"""Repository file scanner with intelligent exclusions and secret filtering."""

import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Set

from devmind.config import DevMindConfig, load_config
from devmind.models import DiscoveredFile
from devmind.utils import (
    compute_file_hash,
    detect_language,
    is_binary_file,
    normalize_path,
)

# Known secret tokens and headers that must never be ingested
SECRET_CONTENT_MARKERS = [
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN PRIVATE KEY",
    "BEGIN CERTIFICATE",
    "BEGIN ENCRYPTED PRIVATE KEY",
    "AIzaSy",  # Google API key prefix
    "ghp_",    # GitHub personal access token prefix
    "sk_live_", # Stripe live secret key
]


class GitIgnoreMatcher:
    """Parses and matches .gitignore rules against file and directory paths."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.patterns: List[str] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        gitignore_path = self.root_dir / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    self.patterns.append(line)
        except Exception:
            pass

    def is_ignored(self, path: Path) -> bool:
        """Check if a relative path matches any .gitignore rule."""
        try:
            rel_path = path.resolve().relative_to(self.root_dir)
            rel_str = rel_path.as_posix()
        except Exception:
            rel_str = path.as_posix()

        parts = rel_str.split("/")

        for pattern in self.patterns:
            clean_pattern = pattern.rstrip("/")
            # Check pattern against the entire relative path or any segment
            if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(rel_str, clean_pattern):
                return True
            for part in parts:
                if fnmatch.fnmatch(part, clean_pattern):
                    return True
        return False


class FileScanner:
    """Recursively scans a project directory for valid source code files."""

    def __init__(self, config: Optional[DevMindConfig] = None):
        self.config = config or load_config()

    def is_secret_file(self, file_path: Path) -> bool:
        """Check if a file should be excluded as a potential secret or credential."""
        name = file_path.name.lower()

        # Allow .env.example or template files
        if name in {".env.example", ".env.template", ".env.sample"}:
            return False

        # Explicitly ban .env and .env.*
        if name == ".env" or name.startswith(".env."):
            return True

        # Match secret file patterns
        for pattern in self.config.secret_patterns:
            if fnmatch.fnmatch(name, pattern.lower()):
                return True

        # Check content for cryptographic private keys or common secrets
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                snippet = f.read(2048)
                for marker in SECRET_CONTENT_MARKERS:
                    if marker in snippet:
                        return True
        except Exception:
            pass

        return False

    def is_ignored_directory(self, dir_name: str) -> bool:
        """Check if a directory name is in the blacklist."""
        return dir_name in self.config.ignore_dirs

    def scan_directory(self, root_dir: str) -> List[DiscoveredFile]:
        """Scan a root directory recursively and return valid discovered source files.

        Applies .gitignore rules, ignore lists, binary checks, and secret filters.
        """
        root_path = Path(root_dir).resolve()
        if not root_path.exists() or not root_path.is_dir():
            return []

        gitignore_matcher = GitIgnoreMatcher(str(root_path))
        discovered: List[DiscoveredFile] = []

        for current_root, dirs, files in os.walk(root_path, topdown=True):
            current_path = Path(current_root)

            # Filter out ignored directories in-place to prevent descending
            dirs[:] = [
                d
                for d in dirs
                if not self.is_ignored_directory(d)
                and not gitignore_matcher.is_ignored(current_path / d)
            ]

            for file_name in files:
                file_path = current_path / file_name

                # Check gitignore
                if gitignore_matcher.is_ignored(file_path):
                    continue

                # Check extension
                suffix = file_path.suffix.lower()
                if suffix not in self.config.supported_extensions:
                    continue

                # Secret check
                if self.is_secret_file(file_path):
                    continue

                # Binary check
                if is_binary_file(str(file_path)):
                    continue

                try:
                    rel_path = normalize_path(file_path.relative_to(root_path))
                    file_hash = compute_file_hash(str(file_path))
                    size = file_path.stat().st_size
                    language = detect_language(str(file_path))

                    discovered.append(
                        DiscoveredFile(
                            file_path=normalize_path(file_path),
                            rel_path=rel_path,
                            file_hash=file_hash,
                            size=size,
                            language=language,
                        )
                    )
                except Exception:
                    # Skip unreadable or corrupted files gracefully
                    continue

        return sorted(discovered, key=lambda f: f.rel_path)
