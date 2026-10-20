"""Safe patch and modification manager with unified diffs, backups, and confirmation safeguards."""

import ast
import difflib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from devmind.config import DevMindConfig, load_config
from devmind.models import PatchProposal
from devmind.utils import safe_read_file


class PatchManager:
    """Safely creates, previews, backs up, and applies code modifications."""

    BACKUP_INDEX_FILENAME = "backup_index.json"

    def __init__(self, config: Optional[DevMindConfig] = None):
        self.config = config or load_config()
        self.backup_dir = Path(self.config.backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backup_index_path(self) -> Path:
        return self.backup_dir / self.BACKUP_INDEX_FILENAME

    def _load_backup_index(self) -> List[Dict[str, Any]]:
        if not self.backup_index_path.exists():
            return []
        try:
            with open(self.backup_index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_backup_index(self, index: List[Dict[str, Any]]) -> None:
        try:
            with open(self.backup_index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass

    def generate_diff(
        self, original_code: str, proposed_code: str, file_label: str = "target_file"
    ) -> str:
        """Generate a clean unified diff between original and proposed code."""
        orig_lines = original_code.splitlines(keepends=True)
        prop_lines = proposed_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            prop_lines,
            fromfile=f"a/{file_label}",
            tofile=f"b/{file_label}",
            lineterm="",
        )
        return "".join(diff)

    def create_backup(self, target_file: str) -> Optional[str]:
        """Create a timestamped backup copy of a file before any mutation."""
        file_path = Path(target_file).resolve()
        if not file_path.exists():
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_filename = f"{timestamp}_{file_path.name}"
        backup_path = self.backup_dir / backup_filename

        try:
            shutil.copy2(file_path, backup_path)
            index = self._load_backup_index()
            entry = {
                "backup_path": str(backup_path),
                "original_file": str(file_path),
                "timestamp": timestamp,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            index.insert(0, entry)
            self._save_backup_index(index)
            return str(backup_path)
        except Exception:
            return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups in reverse chronological order."""
        return self._load_backup_index()

    def create_proposal(
        self,
        target_file: str,
        rel_path: str,
        proposed_code: str,
        explanation: str,
        regression_risks: Optional[List[str]] = None,
        suggested_tests: Optional[List[str]] = None,
    ) -> PatchProposal:
        """Create a structured PatchProposal without modifying any files."""
        orig_code = safe_read_file(target_file) or ""
        diff = self.generate_diff(orig_code, proposed_code, file_label=rel_path)

        return PatchProposal(
            target_file=target_file,
            rel_path=rel_path,
            original_code=orig_code,
            proposed_code=proposed_code,
            unified_diff=diff,
            explanation=explanation,
            regression_risks=regression_risks or [],
            suggested_tests=suggested_tests or [],
        )

    def apply_patch(
        self,
        target_file: str,
        proposed_code: str,
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Safely apply proposed code to target file with required confirmation and backup.

        Returns:
            Dict containing success status, diff, backup path, and any error message.
        """
        file_path = Path(target_file).resolve()
        if not file_path.exists():
            return {
                "success": False,
                "error": f"Target file does not exist: {target_file}",
            }

        orig_code = safe_read_file(str(file_path)) or ""
        diff = self.generate_diff(orig_code, proposed_code, file_label=file_path.name)

        # Enforce explicit user confirmation
        if not confirmed:
            return {
                "success": False,
                "requires_confirmation": True,
                "message": "Modification requires explicit user confirmation before applying.",
                "diff": diff,
            }

        # Check Python syntax if it is a python file
        if file_path.suffix.lower() == ".py":
            try:
                ast.parse(proposed_code, filename=str(file_path))
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Syntax verification failed: {e.msg} at line {e.lineno}",
                    "diff": diff,
                }

        # Create backup
        backup_path = self.create_backup(str(file_path))
        if not backup_path:
            return {
                "success": False,
                "error": "Failed to create safety backup. Modification aborted.",
            }

        # Apply changes
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(proposed_code)
            return {
                "success": True,
                "file_path": str(file_path),
                "backup_path": backup_path,
                "diff": diff,
                "message": f"Successfully applied changes to {file_path.name}. Backup created at {backup_path}.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write changes to file: {str(e)}",
                "backup_path": backup_path,
            }

    def rollback(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Roll back a file to its state from a backup."""
        index = self._load_backup_index()
        if not index:
            return {"success": False, "error": "No backup history available."}

        target_entry = None
        if backup_path:
            for entry in index:
                if entry.get("backup_path") == backup_path:
                    target_entry = entry
                    break
        else:
            target_entry = index[0]  # Most recent

        if not target_entry:
            return {"success": False, "error": f"Backup not found: {backup_path}"}

        backup_file = Path(target_entry["backup_path"])
        orig_file = Path(target_entry["original_file"])

        if not backup_file.exists():
            return {"success": False, "error": f"Backup file missing on disk: {backup_file}"}

        try:
            shutil.copy2(backup_file, orig_file)
            return {
                "success": True,
                "restored_file": str(orig_file),
                "from_backup": str(backup_file),
                "message": f"Successfully rolled back {orig_file.name} from backup.",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to rollback file: {str(e)}"}
