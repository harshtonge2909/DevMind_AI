"""Automated smoke test for the Streamlit web application."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exceptions():
    """Verify that app.py loads and renders the sidebar and main tabs without errors."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert len(at.exception) == 0, f"AppTest encountered exceptions: {[str(e) for e in at.exception]}"
    # Verify 7 tabs
    assert len(at.tabs) == 7
