"""Data models and schemas for DevMind AI."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SourceChunk:
    """Represents an indexed chunk of source code with rich location metadata."""

    content: str
    file_path: str
    rel_path: str
    start_line: int
    end_line: int
    chunk_index: int
    total_chunks: int = 1
    language: str = "python"
    file_hash: str = ""
    chunk_id: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata for ChromaDB storage."""
        return {
            "file_path": self.file_path,
            "rel_path": self.rel_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "language": self.language,
            "file_hash": self.file_hash,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_chroma_doc(cls, content: str, meta: Dict[str, Any]) -> "SourceChunk":
        """Instantiate SourceChunk from Chroma document content and metadata."""
        return cls(
            content=content,
            file_path=meta.get("file_path", ""),
            rel_path=meta.get("rel_path", ""),
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
            chunk_index=int(meta.get("chunk_index", 0)),
            total_chunks=int(meta.get("total_chunks", 1)),
            language=meta.get("language", "unknown"),
            file_hash=meta.get("file_hash", ""),
            chunk_id=meta.get("chunk_id", ""),
        )


@dataclass
class SymbolInfo:
    """Represents a code symbol (function, class, method) extracted via AST."""

    name: str
    symbol_type: str  # function, class, method, async_function
    file_path: str
    rel_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    calls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportInfo:
    """Represents an imported module or name."""

    module: str
    names: List[str]
    alias: Optional[str] = None
    file_path: str = ""
    line_number: int = 1
    is_from: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredFile:
    """Represents a discovered source file during repository scanning."""

    file_path: str
    rel_path: str
    file_hash: str
    size: int
    language: str


@dataclass
class FileMetadata:
    """Metadata for an indexed file."""

    file_path: str
    rel_path: str
    file_hash: str
    size: int
    last_modified: float
    language: str
    chunk_count: int
    symbols: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexManifest:
    """Persistent manifest tracking the indexed state of a project."""

    project_path: str
    indexed_at: str
    embedding_model: str
    total_chunks: int = 0
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with its similarity score."""

    chunk: SourceChunk
    score: float = 0.0


@dataclass
class ReviewFinding:
    """Represents an individual code review finding."""

    title: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # security, bug, performance, readability, maintainability
    file_path: str
    line_number: Optional[int]
    description: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeReviewReport:
    """Aggregated report of code review findings."""

    summary: str
    findings: List[ReviewFinding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    score: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": self.recommendations,
            "score": self.score,
        }


@dataclass
class PatchProposal:
    """Safe proposal for a code modification with diff and safeguards."""

    target_file: str
    rel_path: str
    original_code: str
    proposed_code: str
    unified_diff: str
    explanation: str
    regression_risks: List[str] = field(default_factory=list)
    suggested_tests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    """Standardized output structure for DevMind workflows."""

    workflow_type: str
    success: bool
    answer: str
    sources: List[SourceChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
