"""Utility functions for DevMind AI."""

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional


def extract_text_content(content: Any) -> str:
    """Safely extract plain text from LangChain message content.

    Handles string, list of dicts with 'text' key (common in Gemini responses),
    and list of strings.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts)

    if hasattr(content, "text"):
        return str(content.text)

    return str(content)


def compute_file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute the SHA-256 hash of a text string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_binary_file(file_path: str) -> bool:
    """Check if a file appears to be binary by inspecting initial bytes for null characters."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


def safe_read_file(file_path: str) -> Optional[str]:
    """Read a text file with UTF-8 encoding and fallback to latin-1."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def normalize_path(path: str) -> str:
    """Normalize file path to use forward slashes for cross-platform consistency."""
    return str(Path(path).as_posix())


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".ps1": "powershell",
}


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    suffix = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(suffix, "text")
