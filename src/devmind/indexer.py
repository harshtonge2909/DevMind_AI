"""Codebase indexer with incremental update detection, line-aware chunking, and persistence."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from devmind.code_parser import CodeParser
from devmind.config import DevMindConfig, load_config
from devmind.file_scanner import FileScanner
from devmind.models import DiscoveredFile, IndexManifest, SourceChunk
from devmind.utils import safe_read_file
from devmind.vector_store import VectorStoreManager


class ProjectIndexer:
    """Manages full and incremental indexing of codebases into ChromaDB."""

    MANIFEST_FILENAME = "index_manifest.json"

    def __init__(
        self,
        config: Optional[DevMindConfig] = None,
        vector_store: Optional[VectorStoreManager] = None,
        code_parser: Optional[CodeParser] = None,
    ):
        self.config = config or load_config()
        self.vector_store = vector_store or VectorStoreManager(config=self.config)
        self.code_parser = code_parser or CodeParser()
        self.file_scanner = FileScanner(config=self.config)

    @property
    def manifest_path(self) -> Path:
        """Path to persistent index manifest."""
        data_dir = Path(self.config.data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / self.MANIFEST_FILENAME

    def load_manifest(self) -> Optional[IndexManifest]:
        """Load persistent manifest from disk if it exists."""
        if not self.manifest_path.exists():
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return IndexManifest(
                    project_path=data.get("project_path", ""),
                    indexed_at=data.get("indexed_at", ""),
                    embedding_model=data.get("embedding_model", self.config.embedding_model_name),
                    total_chunks=data.get("total_chunks", 0),
                    files=data.get("files", {}),
                )
        except Exception:
            return None

    def save_manifest(self, manifest: IndexManifest) -> None:
        """Save manifest to disk."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)
        except Exception:
            pass

    def chunk_file(self, discovered_file: DiscoveredFile) -> List[SourceChunk]:
        """Chunk a source file with line number and source-location metadata."""
        content = safe_read_file(discovered_file.file_path)
        if not content or not content.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        raw_chunks = splitter.split_text(content)
        if not raw_chunks:
            return []

        chunks: List[SourceChunk] = []
        total = len(raw_chunks)
        search_start = 0

        for idx, chunk_text in enumerate(raw_chunks):
            pos = content.find(chunk_text, search_start)
            if pos == -1:
                pos = content.find(chunk_text)

            if pos == -1:
                start_line = 1
                end_line = max(1, len(chunk_text.splitlines()))
            else:
                start_line = content[:pos].count("\n") + 1
                end_line = start_line + chunk_text.count("\n")
                search_start = max(0, pos)

            chunk_id = f"{discovered_file.rel_path}::c{idx}"
            chunks.append(
                SourceChunk(
                    content=chunk_text,
                    file_path=discovered_file.file_path,
                    rel_path=discovered_file.rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_index=idx,
                    total_chunks=total,
                    language=discovered_file.language,
                    file_hash=discovered_file.file_hash,
                    chunk_id=chunk_id,
                )
            )

        return chunks

    def detect_changes(
        self, discovered_files: List[DiscoveredFile], manifest: Optional[IndexManifest]
    ) -> Tuple[List[DiscoveredFile], List[DiscoveredFile], List[str]]:
        """Identify added, modified, and deleted files compared to existing manifest.

        Returns:
            Tuple of (added_files, modified_files, deleted_rel_paths)
        """
        if not manifest or not manifest.files:
            return discovered_files, [], []

        current_by_rel = {f.rel_path: f for f in discovered_files}
        manifest_files = manifest.files

        added: List[DiscoveredFile] = []
        modified: List[DiscoveredFile] = []
        deleted: List[str] = []

        # Find added and modified
        for rel_path, disc in current_by_rel.items():
            if rel_path not in manifest_files:
                added.append(disc)
            else:
                prev = manifest_files[rel_path]
                if prev.get("file_hash") != disc.file_hash:
                    modified.append(disc)

        # Find deleted
        for rel_path in manifest_files:
            if rel_path not in current_by_rel:
                deleted.append(rel_path)

        return added, modified, deleted

    def index_project(
        self,
        project_path: str,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Index or incrementally update the project codebase.

        Args:
            project_path: Target directory to scan and index.
            force_reindex: If True, resets index and rebuilds from scratch.
            progress_callback: Optional callback(current, total, message).
        """
        project_dir = Path(project_path).resolve()
        if not project_dir.exists():
            return {
                "success": False,
                "error": f"Project directory does not exist: {project_path}",
                "indexed_files": 0,
                "total_chunks": 0,
            }

        if progress_callback:
            progress_callback(0, 100, f"Scanning {project_path} for source files...")

        discovered_files = self.file_scanner.scan_directory(str(project_dir))

        if not discovered_files:
            return {
                "success": True,
                "warning": f"No supported source code files found in {project_path}.",
                "indexed_files": 0,
                "total_chunks": 0,
            }

        manifest = None if force_reindex else self.load_manifest()

        if force_reindex:
            if progress_callback:
                progress_callback(10, 100, "Resetting vector database...")
            self.vector_store.reset()
            manifest_files: Dict[str, Dict[str, Any]] = {}
            added, modified, deleted = discovered_files, [], []
        else:
            manifest_files = manifest.files if manifest else {}
            added, modified, deleted = self.detect_changes(discovered_files, manifest)

        # Handle deleted files
        for del_path in deleted:
            if progress_callback:
                progress_callback(20, 100, f"Removing deleted file: {del_path}")
            old_data = manifest_files.pop(del_path, {})
            old_chunk_ids = old_data.get("chunk_ids", [])
            if old_chunk_ids:
                self.vector_store.delete_chunks(old_chunk_ids)
            else:
                self.vector_store.delete_file_chunks(del_path)

        # Handle modified files - delete previous chunks
        for mod_file in modified:
            old_data = manifest_files.get(mod_file.rel_path, {})
            old_chunk_ids = old_data.get("chunk_ids", [])
            if old_chunk_ids:
                self.vector_store.delete_chunks(old_chunk_ids)
            else:
                self.vector_store.delete_file_chunks(mod_file.rel_path)

        files_to_process = added + modified
        total_files = len(files_to_process)
        new_chunks: List[SourceChunk] = []

        # Parse AST for all discovered files so code parser has full symbol map
        for disc in discovered_files:
            self.code_parser.parse_file(disc.file_path, disc.rel_path)

        # Chunk and prepare vectors for added and modified files
        for i, file_info in enumerate(files_to_process):
            if progress_callback:
                pct = 20 + int((i / max(1, total_files)) * 60)
                progress_callback(pct, 100, f"Chunking {file_info.rel_path} ({i+1}/{total_files})")

            file_chunks = self.chunk_file(file_info)
            new_chunks.extend(file_chunks)

            parsed_data = self.code_parser.parsed_files.get(file_info.rel_path, {})
            symbols_dicts = [s.to_dict() for s in parsed_data.get("symbols", [])]

            manifest_files[file_info.rel_path] = {
                "file_path": file_info.file_path,
                "rel_path": file_info.rel_path,
                "file_hash": file_info.file_hash,
                "size": file_info.size,
                "language": file_info.language,
                "chunk_ids": [c.chunk_id for c in file_chunks],
                "chunk_count": len(file_chunks),
                "symbols_count": len(symbols_dicts),
            }

        # Embed and insert chunks into ChromaDB
        if new_chunks:
            if progress_callback:
                progress_callback(85, 100, f"Generating embeddings for {len(new_chunks)} chunks...")
            self.vector_store.add_chunks(new_chunks)

        total_stored_chunks = sum(
            f.get("chunk_count", 0) for f in manifest_files.values()
        )

        new_manifest = IndexManifest(
            project_path=str(project_dir),
            indexed_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=self.config.embedding_model_name,
            total_chunks=total_stored_chunks,
            files=manifest_files,
        )
        self.save_manifest(new_manifest)

        if progress_callback:
            progress_callback(100, 100, "Indexing complete!")

        return {
            "success": True,
            "project_path": str(project_dir),
            "total_files": len(discovered_files),
            "added_files": len(added),
            "modified_files": len(modified),
            "deleted_files": len(deleted),
            "total_chunks": total_stored_chunks,
            "new_chunks_added": len(new_chunks),
        }
