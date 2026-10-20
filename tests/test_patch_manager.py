"""Tests for PatchManager and safe code modification safeguards."""

from devmind.config import DevMindConfig
from devmind.patch_manager import PatchManager


def test_unified_diff_generation():
    pm = PatchManager()
    orig = "def add(a, b):\n    return a + b\n"
    prop = "def add(a: int, b: int) -> int:\n    return a + b\n"

    diff = pm.generate_diff(orig, prop, "utils.py")
    assert "--- a/utils.py" in diff
    assert "+++ b/utils.py" in diff
    assert "+def add(a: int, b: int) -> int:" in diff
    assert "-def add(a, b):" in diff


def test_patch_requires_confirmation(tmp_path):
    target = tmp_path / "test.py"
    target.write_text("x = 1\n", encoding="utf-8")

    pm = PatchManager(config=DevMindConfig(backup_dir=str(tmp_path / "backups")))
    res = pm.apply_patch(str(target), "x = 2\n", confirmed=False)

    assert res["success"] is False
    assert res["requires_confirmation"] is True
    # File content must NOT have changed
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_patch_validates_python_syntax(tmp_path):
    target = tmp_path / "test.py"
    target.write_text("x = 1\n", encoding="utf-8")

    pm = PatchManager(config=DevMindConfig(backup_dir=str(tmp_path / "backups")))
    # Broken syntax
    res = pm.apply_patch(str(target), "def broken(:\n    pass\n", confirmed=True)

    assert res["success"] is False
    assert "Syntax verification failed" in res["error"]
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_patch_creates_backup_and_rolls_back(tmp_path):
    target = tmp_path / "test.py"
    target.write_text("original_content = True\n", encoding="utf-8")

    pm = PatchManager(config=DevMindConfig(backup_dir=str(tmp_path / "backups")))
    res = pm.apply_patch(str(target), "original_content = False\n", confirmed=True)

    assert res["success"] is True
    assert target.read_text(encoding="utf-8") == "original_content = False\n"
    assert "backup_path" in res

    # Rollback
    rb_res = pm.rollback()
    assert rb_res["success"] is True
    assert target.read_text(encoding="utf-8") == "original_content = True\n"
