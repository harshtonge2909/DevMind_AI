"""Pytest configuration and shared fixtures for DevMind AI test suite."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure src/ is on python path for all tests
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from devmind.config import DevMindConfig
from devmind.models import SourceChunk


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary directory structure mimicking a Python repository."""
    proj = tmp_path / "sample_project"
    proj.mkdir()

    # Main file
    (proj / "main.py").write_text(
        "from utils import add, Calculator\n\nresult = add(10, 20)\nprint(result)\n",
        encoding="utf-8",
    )

    # Utils file
    (proj / "utils.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers together."""\n'
        "    return a + b\n\n"
        "class Calculator:\n"
        '    """Basic arithmetic calculator."""\n'
        "    def multiply(self, x, y):\n"
        "        return x * y\n",
        encoding="utf-8",
    )

    # Secret files that must be excluded
    (proj / ".env").write_text("Gemini_API_Key=secret_test_key_12345\n", encoding="utf-8")
    (proj / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEA0\n-----END RSA PRIVATE KEY-----\n", encoding="utf-8")
    (proj / ".env.example").write_text("Gemini_API_Key=your_key_here\n", encoding="utf-8")

    # Ignored directory
    venv = proj / ".venv"
    venv.mkdir()
    (venv / "site.py").write_text("# ignored venv file\n", encoding="utf-8")

    # .gitignore
    (proj / ".gitignore").write_text("temp_log.txt\nignored_folder/\n", encoding="utf-8")
    (proj / "temp_log.txt").write_text("temporary log content\n", encoding="utf-8")

    return proj


@pytest.fixture
def mock_embeddings():
    """Mock embeddings that return deterministic 3072-dimensional vectors without network calls."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[0.1] * 3072 for _ in texts]
    mock.embed_query.side_effect = lambda query: [0.1] * 3072
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM returning deterministic AIMessage responses without network calls."""
    mock = MagicMock()
    response = MagicMock()
    response.content = "DevMind AI mock answer grounded in test code."
    mock.invoke.return_value = response
    return mock
