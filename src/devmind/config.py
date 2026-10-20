"""Configuration and environment management for DevMind AI."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set
from dotenv import load_dotenv


_DOTENV_LOADED = False


def ensure_environment(reload_dotenv: bool = False) -> Optional[str]:
    """Load environment variables and ensure Google API key is configured.

    Checks Gemini_API_Key, GEMINI_API_KEY, and GOOGLE_API_KEY in order.
    Sets GOOGLE_API_KEY in os.environ for library compatibility.
    Never prints or logs the raw key value.
    """
    global _DOTENV_LOADED
    if not _DOTENV_LOADED or reload_dotenv:
        load_dotenv()
        _DOTENV_LOADED = True

    api_key = (
        os.getenv("Gemini_API_Key")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if api_key:
        api_key = api_key.strip()
        # Ensure GOOGLE_API_KEY is populated for langchain-google-genai
        os.environ["GOOGLE_API_KEY"] = api_key
        return api_key

    return None


@dataclass
class DevMindConfig:
    """Central configuration for DevMind AI."""

    # Models (Preserve working verified models)
    model_name: str = "gemini-3.6-flash"
    embedding_model_name: str = "gemini-embedding-001"
    temperature: float = 0.2
    max_output_tokens: Optional[int] = None

    # Chunking & Retrieval
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_top_k: int = 4

    # Storage paths
    chroma_dir: str = "./chroma_db"
    data_dir: str = "./data"
    backup_dir: str = "./.devmind_backups"

    # Supported source extensions
    supported_extensions: Set[str] = field(
        default_factory=lambda: {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".html",
            ".css",
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".sql",
            ".sh",
            ".ps1",
        }
    )

    # Directories to ignore
    ignore_dirs: Set[str] = field(
        default_factory=lambda: {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            "wheels",
            ".egg-info",
            ".idea",
            ".vscode",
            "chroma_db",
            "data",
            ".devmind_backups",
            ".gemini",
            ".system_generated",
            "site-packages",
        }
    )

    # Secret filenames and patterns to never ingest
    secret_patterns: List[str] = field(
        default_factory=lambda: [
            ".env",
            "*.pem",
            "*.key",
            "*.pfx",
            "*.p12",
            "id_rsa*",
            "id_ed25519*",
            "credentials.json",
            "service_account.json",
            "token.json",
            "secrets.yaml",
            "secrets.json",
        ]
    )

    def is_api_key_set(self) -> bool:
        """Check if the API key is set without exposing its value."""
        return ensure_environment() is not None

    def get_api_key(self) -> Optional[str]:
        """Retrieve the configured API key securely."""
        return ensure_environment()


def load_config(**overrides) -> DevMindConfig:
    """Load configuration with optional runtime overrides."""
    ensure_environment()
    config = DevMindConfig()
    for key, val in overrides.items():
        if hasattr(config, key) and val is not None:
            setattr(config, key, val)
    return config
