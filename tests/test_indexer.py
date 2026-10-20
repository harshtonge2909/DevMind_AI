"""Tests for ProjectIndexer, line-aware chunking, and incremental updates."""

from unittest.mock import MagicMock
from devmind.config import DevMindConfig
from devmind.file_scanner import DiscoveredFile
from devmind.indexer import ProjectIndexer
from devmind.models import IndexManifest


def test_chunking_preserves_line_numbers_and_metadata(tmp_path):
    config = DevMindConfig(chunk_size=100, chunk_overlap=20)
    indexer = ProjectIndexer(config=config, vector_store=MagicMock())

    sample_file = tmp_path / "sample.py"
    sample_file.write_text(
        "# Line 1\n"
        "# Line 2\n"
        "def foo():\n"
        "    return 42\n"
        "\n"
        "# Line 6\n"
        "def bar():\n"
        "    return 99\n",
        encoding="utf-8",
    )

    disc = DiscoveredFile(
        file_path=str(sample_file),
        rel_path="sample.py",
        file_hash="fakehash123",
        size=sample_file.stat().st_size,
        language="python",
    )

    chunks = indexer.chunk_file(disc)
    assert len(chunks) >= 1

    first_chunk = chunks[0]
    assert first_chunk.rel_path == "sample.py"
    assert first_chunk.start_line == 1
    assert first_chunk.end_line >= 1
    assert first_chunk.chunk_id == "sample.py::c0"
    assert first_chunk.file_hash == "fakehash123"


def test_detect_changes_added_modified_deleted():
    indexer = ProjectIndexer(vector_store=MagicMock())

    current_files = [
        DiscoveredFile("path/a.py", "a.py", "hash_new", 100, "python"),  # Modified
        DiscoveredFile("path/b.py", "b.py", "hash_same", 100, "python"), # Same
        DiscoveredFile("path/c.py", "c.py", "hash_added", 100, "python"), # Added
    ]

    manifest = IndexManifest(
        project_path="path",
        indexed_at="2026-01-01T00:00:00",
        embedding_model="gemini-embedding-001",
        files={
            "a.py": {"file_hash": "hash_old", "chunk_ids": ["a.py::c0"]},
            "b.py": {"file_hash": "hash_same", "chunk_ids": ["b.py::c0"]},
            "d.py": {"file_hash": "hash_deleted", "chunk_ids": ["d.py::c0"]}, # Deleted
        },
    )

    added, modified, deleted = indexer.detect_changes(current_files, manifest)

    assert len(added) == 1
    assert added[0].rel_path == "c.py"

    assert len(modified) == 1
    assert modified[0].rel_path == "a.py"

    assert deleted == ["d.py"]


def test_index_project_offline_with_mock_vector_store(temp_project_dir, tmp_path):
    mock_store = MagicMock()
    mock_store.add_chunks.return_value = 5

    config = DevMindConfig(data_dir=str(tmp_path / "data"))
    indexer = ProjectIndexer(config=config, vector_store=mock_store)

    result = indexer.index_project(str(temp_project_dir))

    assert result["success"] is True
    assert result["total_files"] >= 2
    assert mock_store.add_chunks.called

    # Verify manifest was saved
    loaded_manifest = indexer.load_manifest()
    assert loaded_manifest is not None
    assert "main.py" in loaded_manifest.files
    assert "utils.py" in loaded_manifest.files
