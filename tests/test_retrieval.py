"""Tests for CodebaseRetriever and citation formatting."""

from unittest.mock import MagicMock
from devmind.code_parser import CodeParser
from devmind.models import RetrievalResult, SourceChunk
from devmind.retrieval import CodebaseRetriever


def test_format_evidence_prompt():
    chunks = [
        RetrievalResult(
            chunk=SourceChunk(
                content="def add(a, b):\n    return a + b",
                file_path="project/utils.py",
                rel_path="project/utils.py",
                start_line=1,
                end_line=2,
                chunk_index=0,
                chunk_id="c0",
            ),
            score=0.95,
        )
    ]

    evidence = CodebaseRetriever.format_evidence_prompt(chunks)
    assert "--- Source Snippet 1 [project/utils.py : Lines 1-2] ---" in evidence
    assert "def add(a, b):" in evidence


def test_format_citations():
    chunks = [
        RetrievalResult(
            chunk=SourceChunk(
                content="def add(a, b):\n    return a + b",
                file_path="project/utils.py",
                rel_path="project/utils.py",
                start_line=1,
                end_line=2,
                chunk_index=0,
                chunk_id="c0",
            ),
            score=0.88888,
        )
    ]

    citations = CodebaseRetriever.format_citations(chunks)
    assert len(citations) == 1
    assert citations[0]["rel_path"] == "project/utils.py"
    assert citations[0]["lines"] == "L1-L2"
    assert citations[0]["score"] == 0.8889


def test_retrieve_by_symbol(temp_project_dir):
    parser = CodeParser()
    parser.parse_file(str(temp_project_dir / "utils.py"), "utils.py")

    retriever = CodebaseRetriever(vector_store=MagicMock(), code_parser=parser)
    results = retriever.retrieve_by_symbol("add")

    assert len(results) >= 1
    assert "def add" in results[0].chunk.content
    assert results[0].chunk.start_line == 1
