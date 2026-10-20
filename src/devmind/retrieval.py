"""Retrieval engine combining vector similarity search with AST symbol grounding."""

from typing import Any, Dict, List, Optional

from devmind.code_parser import CodeParser
from devmind.config import DevMindConfig, load_config
from devmind.models import RetrievalResult, SourceChunk
from devmind.utils import safe_read_file
from devmind.vector_store import VectorStoreManager


class CodebaseRetriever:
    """Retrieves relevant code chunks using hybrid semantic and structural symbol queries."""

    def __init__(
        self,
        config: Optional[DevMindConfig] = None,
        vector_store: Optional[VectorStoreManager] = None,
        code_parser: Optional[CodeParser] = None,
    ):
        self.config = config or load_config()
        self.vector_store = vector_store or VectorStoreManager(config=self.config)
        self.code_parser = code_parser or CodeParser()

    def retrieve(
        self, query: str, top_k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform semantic retrieval from Chroma vector store."""
        k = top_k or self.config.default_top_k
        return self.vector_store.similarity_search(query=query, k=k, filter_dict=filter_dict)

    def retrieve_by_symbol(self, symbol_name: str) -> List[RetrievalResult]:
        """Look up symbol definition from AST and construct an exact source chunk."""
        symbols = self.code_parser.find_symbols(symbol_name, exact=True)
        if not symbols:
            # Try partial matching
            symbols = self.code_parser.find_symbols(symbol_name, exact=False)

        results: List[RetrievalResult] = []
        for sym in symbols[:3]:
            content = safe_read_file(sym.file_path)
            if content:
                lines = content.splitlines()
                # 1-indexed to 0-indexed slice
                start_idx = max(0, sym.start_line - 1)
                end_idx = min(len(lines), sym.end_line)
                snippet = "\n".join(lines[start_idx:end_idx])

                chunk = SourceChunk(
                    content=snippet,
                    file_path=sym.file_path,
                    rel_path=sym.rel_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    chunk_index=0,
                    total_chunks=1,
                    language="python",
                    chunk_id=f"ast::{sym.name}",
                )
                results.append(RetrievalResult(chunk=chunk, score=1.0))

        return results

    def hybrid_retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Combine exact symbol matches with vector similarity search."""
        k = top_k or self.config.default_top_k

        # 1. Check if query mentions a symbol or function name
        symbol_results: List[RetrievalResult] = []
        words = [w.strip("(),'\"`[]{}") for w in query.split()]
        for word in words:
            if len(word) >= 3 and (word.isidentifier() or "." in word):
                exact_syms = self.retrieve_by_symbol(word)
                for res in exact_syms:
                    if not any(r.chunk.chunk_id == res.chunk.chunk_id for r in symbol_results):
                        symbol_results.append(res)

        # 2. Vector similarity search
        vector_results = self.retrieve(query=query, top_k=k)

        # Merge, prioritizing symbol matches and removing duplicates
        combined: List[RetrievalResult] = []
        seen_keys = set()

        for res in symbol_results + vector_results:
            key = (res.chunk.rel_path, res.chunk.start_line, res.chunk.end_line)
            if key not in seen_keys:
                seen_keys.add(key)
                combined.append(res)
                if len(combined) >= k:
                    break

        return combined

    @staticmethod
    def format_evidence_prompt(results: List[RetrievalResult]) -> str:
        """Format retrieved code chunks into structured evidence for LLM prompt."""
        if not results:
            return "No relevant code snippets found in repository."

        parts: List[str] = []
        for i, res in enumerate(results, 1):
            chunk = res.chunk
            header = f"--- Source Snippet {i} [{chunk.rel_path} : Lines {chunk.start_line}-{chunk.end_line}] ---"
            parts.append(f"{header}\n{chunk.content}\n")

        return "\n".join(parts)

    @staticmethod
    def format_citations(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Extract user-facing citations list from retrieval results."""
        citations: List[Dict[str, Any]] = []
        for res in results:
            chunk = res.chunk
            citations.append(
                {
                    "rel_path": chunk.rel_path,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "lines": f"L{chunk.start_line}-L{chunk.end_line}",
                    "content": chunk.content,
                    "score": round(res.score, 4),
                }
            )
        return citations
