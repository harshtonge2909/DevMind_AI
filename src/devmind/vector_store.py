"""ChromaDB vector store integration with persistence and deduplication."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from devmind.config import DevMindConfig, load_config
from devmind.models import RetrievalResult, SourceChunk


class VectorStoreManager:
    """Manages ChromaDB vector storage, chunk insertion, deletion, and queries."""

    COLLECTION_NAME = "devmind_codebase"

    def __init__(
        self,
        config: Optional[DevMindConfig] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.config = config or load_config()
        self._embeddings = embeddings
        self._store: Optional[Chroma] = None

    @property
    def embeddings(self) -> Embeddings:
        """Lazily initialize embeddings using the configured Google model."""
        if self._embeddings is None:
            api_key = self.config.get_api_key()
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model_name,
                google_api_key=api_key,
            )
        return self._embeddings

    @property
    def store(self) -> Chroma:
        """Lazily initialize the persistent Chroma store."""
        if self._store is None:
            chroma_path = Path(self.config.chroma_dir).resolve()
            chroma_path.mkdir(parents=True, exist_ok=True)

            self._store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(chroma_path),
            )
        return self._store

    def add_chunks(self, chunks: List[SourceChunk]) -> int:
        """Add chunks to vector store with deterministic chunk IDs to avoid duplicates."""
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Use batching for large chunk sets (Chroma handles batches well)
        batch_size = 64
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            total_added += len(batch_texts)

        return total_added

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete specific chunks by their deterministic IDs."""
        if not chunk_ids:
            return
        try:
            self.store.delete(ids=chunk_ids)
        except Exception:
            pass

    def delete_file_chunks(self, rel_path: str) -> None:
        """Delete all chunks associated with a relative file path."""
        try:
            # Query the underlying collection for matching IDs
            collection = self.store._collection
            results = collection.get(where={"rel_path": rel_path})
            matching_ids = results.get("ids", [])
            if matching_ids:
                collection.delete(ids=matching_ids)
        except Exception:
            pass

    def similarity_search(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform similarity search and return SourceChunk results with scores."""
        try:
            results = self.store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter_dict,
            )
        except Exception:
            # Fallback to standard similarity search if relevance scores fail
            docs = self.store.similarity_search(query=query, k=k, filter=filter_dict)
            results = [(doc, 0.0) for doc in docs]

        retrieval_results: List[RetrievalResult] = []
        for doc, score in results:
            chunk = SourceChunk.from_chroma_doc(
                content=doc.page_content,
                meta=doc.metadata,
            )
            retrieval_results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return retrieval_results

    def count(self) -> int:
        """Return total number of stored vectors."""
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Reset the vector store collection."""
        try:
            self.store.reset_collection()
        except Exception:
            try:
                self.store._collection.delete()
            except Exception:
                pass
