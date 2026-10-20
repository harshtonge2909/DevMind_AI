"""Tests for FileScanner and GitIgnoreMatcher."""

from pathlib import Path
from devmind.file_scanner import FileScanner, GitIgnoreMatcher
from devmind.config import DevMindConfig


def test_scanner_finds_valid_source_files(temp_project_dir):
    scanner = FileScanner()
    discovered = scanner.scan_directory(str(temp_project_dir))

    discovered_rel_paths = [f.rel_path for f in discovered]
    assert "main.py" in discovered_rel_paths
    assert "utils.py" in discovered_rel_paths

    # Verify .env.example is not flagged as a secret file
    assert not scanner.is_secret_file(temp_project_dir / ".env.example")


def test_scanner_excludes_secrets_and_env(temp_project_dir):
    scanner = FileScanner()
    discovered = scanner.scan_directory(str(temp_project_dir))

    discovered_rel_paths = [f.rel_path for f in discovered]
    # .env must be excluded
    assert ".env" not in discovered_rel_paths
    # Private key file must be excluded
    assert "id_rsa" not in discovered_rel_paths


def test_scanner_excludes_venv_and_ignored_directories(temp_project_dir):
    scanner = FileScanner()
    discovered = scanner.scan_directory(str(temp_project_dir))

    discovered_rel_paths = [f.rel_path for f in discovered]
    for p in discovered_rel_paths:
        assert ".venv" not in p
        assert not p.startswith(".venv/")


def test_scanner_respects_gitignore(temp_project_dir):
    scanner = FileScanner()
    discovered = scanner.scan_directory(str(temp_project_dir))

    discovered_rel_paths = [f.rel_path for f in discovered]
    assert "temp_log.txt" not in discovered_rel_paths


def test_binary_file_exclusion(tmp_path):
    bin_file = tmp_path / "compiled.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\x04\x05binary_data")

    scanner = FileScanner()
    discovered = scanner.scan_directory(str(tmp_path))
    assert len(discovered) == 0


def test_empty_directory_handling(tmp_path):
    scanner = FileScanner()
    discovered = scanner.scan_directory(str(tmp_path))
    assert discovered == []


def test_nonexistent_directory_handling():
    scanner = FileScanner()
    discovered = scanner.scan_directory("non_existent_folder_xyz_123")
    assert discovered == []
