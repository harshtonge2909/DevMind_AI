"""Tests for WorkflowOrchestrator and multi-step engineering workflows."""

from unittest.mock import MagicMock
import pytest
from devmind.code_review import CodeReviewer, StaticCodeAuditor
from devmind.config import DevMindConfig
from devmind.workflows import WorkflowOrchestrator


def test_missing_api_key_raises_informative_error(monkeypatch):
    monkeypatch.delenv("Gemini_API_Key", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    config = DevMindConfig()
    orchestrator = WorkflowOrchestrator(config=config)

    with pytest.raises(ValueError, match="Google API Key is not configured"):
        _ = orchestrator.llm


def test_ask_workflow_input_validation():
    orchestrator = WorkflowOrchestrator(llm=MagicMock())
    res = orchestrator.ask("")
    assert res.success is False
    assert "cannot be empty" in res.answer


def test_explain_workflow_input_validation():
    orchestrator = WorkflowOrchestrator(llm=MagicMock())
    res = orchestrator.explain("   ")
    assert res.success is False
    assert "cannot be empty" in res.answer


def test_debug_workflow_input_validation():
    orchestrator = WorkflowOrchestrator(llm=MagicMock())
    res = orchestrator.debug("")
    assert res.success is False
    assert "cannot be empty" in res.answer


def test_ask_workflow_with_mock_llm(mock_llm):
    mock_retriever = MagicMock()
    mock_retriever.hybrid_retrieve.return_value = []
    mock_retriever.format_evidence_prompt.return_value = "No code found."

    orchestrator = WorkflowOrchestrator(llm=mock_llm)
    orchestrator.retriever = mock_retriever

    result = orchestrator.ask("How does this project work?")
    assert result.success is True
    assert "DevMind AI mock answer" in result.answer
    assert mock_llm.invoke.called


def test_static_code_auditor_detects_issues():
    auditor = StaticCodeAuditor()
    sample_code = (
        'api_key = "AIzaSyD_fake_secret_key_12345"\n'
        "def bad_func(items=[]):\n"
        "    try:\n"
        "        eval('2 + 2')\n"
        "    except:\n"
        "        pass\n"
    )

    findings = auditor.analyze_file("vulnerable.py", sample_code)
    finding_titles = [f.title for f in findings]

    assert "Possible Hardcoded Secret" in finding_titles
    assert "Mutable Default Argument" in finding_titles
    assert "Dynamic Code Execution (eval/exec)" in finding_titles
    assert "Bare Exception Clause" in finding_titles


def test_review_workflow_offline():
    reviewer = CodeReviewer()
    report = reviewer.review_file("project/utils.py")
    assert report is not None
    assert isinstance(report.findings, list)
    assert 0 <= report.score <= 100


def test_llm_retry_on_transient_error():
    mock_llm = MagicMock()
    # First call raises transient 503, second succeeds
    response = MagicMock()
    response.content = "Recovered response after transient error"
    mock_llm.invoke.side_effect = [
        RuntimeError("503 Service Unavailable"),
        response,
    ]

    orchestrator = WorkflowOrchestrator(llm=mock_llm)
    out = orchestrator._invoke_with_retry(["test message"], max_retries=2)
    assert out == "Recovered response after transient error"
    assert mock_llm.invoke.call_count == 2
