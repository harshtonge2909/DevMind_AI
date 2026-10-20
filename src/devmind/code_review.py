"""Code review engine combining static heuristic analysis with LLM deep inspection."""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from devmind.config import DevMindConfig, load_config
from devmind.models import CodeReviewReport, ReviewFinding
from devmind.prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from devmind.utils import extract_text_content, safe_read_file


class StaticCodeAuditor:
    """Performs static heuristic checks on Python source code."""

    SECRET_REGEX = re.compile(
        r"""(?i)(password|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\s*=\s*['"][a-zA-Z0-9_\-\.]{8,}['"]"""
    )
    SHELL_INJECTION_REGEX = re.compile(
        r"""subprocess\.(run|call|Popen)\(.*?shell\s*=\s*True"""
    )

    def analyze_file(self, file_path: str, content: str) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []
        lines = content.splitlines()

        # Regex-based scans
        for lineno, line in enumerate(lines, 1):
            # Hardcoded secrets
            if self.SECRET_REGEX.search(line):
                findings.append(
                    ReviewFinding(
                        title="Possible Hardcoded Secret",
                        severity="CRITICAL",
                        category="security",
                        file_path=file_path,
                        line_number=lineno,
                        description=f"Line contains pattern resembling an embedded credential or token.",
                        suggestion="Extract secrets to environment variables or a secure secret manager.",
                    )
                )

            # Insecure shell execution
            if self.SHELL_INJECTION_REGEX.search(line):
                findings.append(
                    ReviewFinding(
                        title="Insecure Subprocess Execution",
                        severity="HIGH",
                        category="security",
                        file_path=file_path,
                        line_number=lineno,
                        description="Subprocess invocation with `shell=True` can lead to command injection.",
                        suggestion="Pass arguments as a list and avoid `shell=True`.",
                    )
                )

            # Insecure eval / exec
            if "eval(" in line or "exec(" in line:
                findings.append(
                    ReviewFinding(
                        title="Dynamic Code Execution (eval/exec)",
                        severity="HIGH",
                        category="security",
                        file_path=file_path,
                        line_number=lineno,
                        description="Direct use of `eval` or `exec` can allow arbitrary code execution.",
                        suggestion="Refactor using `ast.literal_eval` or safe domain-specific parsing.",
                    )
                )

            # Bare except
            if re.match(r"^\s*except\s*:", line):
                findings.append(
                    ReviewFinding(
                        title="Bare Exception Clause",
                        severity="MEDIUM",
                        category="maintainability",
                        file_path=file_path,
                        line_number=lineno,
                        description="Bare `except:` catches SystemExit and KeyboardInterrupt.",
                        suggestion="Specify the exact exception type (e.g., `except Exception:` or specific error).",
                    )
                )

        # AST-based checks for Python files
        if Path(file_path).suffix.lower() == ".py":
            try:
                tree = ast.parse(content, filename=file_path)
                for node in ast.walk(tree):
                    # Mutable default arguments
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for default in node.args.defaults + node.args.kw_defaults:
                            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                                findings.append(
                                    ReviewFinding(
                                        title="Mutable Default Argument",
                                        severity="MEDIUM",
                                        category="bug",
                                        file_path=file_path,
                                        line_number=getattr(node, "lineno", None),
                                        description=f"Function `{node.name}` uses a mutable default argument.",
                                        suggestion="Use `None` as default and assign a new container inside the function.",
                                    )
                                )

                    # Missing docstrings in public functions
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        if not ast.get_docstring(node):
                            findings.append(
                                ReviewFinding(
                                    title="Missing Function Docstring",
                                    severity="LOW",
                                    category="readability",
                                    file_path=file_path,
                                    line_number=getattr(node, "lineno", None),
                                    description=f"Public function `{node.name}` is missing a docstring.",
                                    suggestion="Add a docstring explaining arguments, behavior, and return value.",
                                )
                            )
            except SyntaxError:
                pass

        return findings


class CodeReviewer:
    """Coordinates static analysis and LLM-assisted deep code review."""

    def __init__(self, config: Optional[DevMindConfig] = None, llm=None):
        self.config = config or load_config()
        self.llm = llm
        self.auditor = StaticCodeAuditor()

    def review_file(self, file_path: str, rel_path: str = "") -> CodeReviewReport:
        """Run comprehensive review on target file."""
        content = safe_read_file(file_path)
        if not content:
            return CodeReviewReport(
                summary=f"Unable to read file: {file_path}",
                findings=[],
                recommendations=["Ensure the file exists and is readable."],
            )

        rel_label = rel_path or Path(file_path).name
        static_findings = self.auditor.analyze_file(rel_label, content)

        # Format static findings for LLM prompt
        if static_findings:
            static_text = "\n".join(
                f"- [{f.severity}] {f.title} (Line {f.line_number}): {f.description}"
                for f in static_findings
            )
        else:
            static_text = "No static heuristic issues detected."

        # If LLM is available, perform deep inspection
        llm_summary = ""
        llm_recommendations: List[str] = []

        if self.llm:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(content=REVIEW_SYSTEM_PROMPT),
                    HumanMessage(
                        content=REVIEW_USER_PROMPT.format(
                            file_path=rel_label,
                            source_code=content[:8000],  # Bounded window
                            static_findings=static_text,
                        )
                    ),
                ]
                response = self.llm.invoke(messages)
                llm_summary = extract_text_content(response.content)

                # Parse LLM findings from text if present
                self._parse_llm_findings(llm_summary, static_findings, rel_label)
            except Exception as e:
                llm_summary = (
                    f"Static analysis completed. LLM deep review could not be completed: {str(e)}"
                )
        else:
            llm_summary = (
                f"Static analysis completed ({len(static_findings)} issue(s) detected). "
                "LLM review is unavailable without API credentials."
            )

        # Compute quality score (100 base, deductions per finding)
        deductions = {
            "CRITICAL": 30,
            "HIGH": 20,
            "MEDIUM": 10,
            "LOW": 5,
            "INFO": 0,
        }
        score = 100
        for f in static_findings:
            score -= deductions.get(f.severity.upper(), 5)
        score = max(0, min(100, score))

        recommendations = [
            "Address all CRITICAL and HIGH severity findings immediately.",
            "Write comprehensive unit tests covering identified boundary conditions.",
            "Ensure docstrings and type annotations follow PEP 8 and PEP 484.",
        ]

        return CodeReviewReport(
            summary=llm_summary,
            findings=static_findings,
            recommendations=recommendations,
            score=score,
        )

    def _parse_llm_findings(
        self, text: str, existing_findings: List[ReviewFinding], file_label: str
    ) -> None:
        """Parse structured findings from LLM output to supplement static findings."""
        pattern = re.compile(
            r"\[SEVERITY\]\s*[:\-]?\s*([A-Z]+).*?"
            r"\[CATEGORY\]\s*[:\-]?\s*([A-Za-z]+).*?"
            r"(?:\[LOCATION\]\s*[:\-]?\s*(?:.*?(?:line\s*(\d+))?))?.*?"
            r"\[DESCRIPTION\]\s*[:\-]?\s*(.*?)(?=\[SUGGESTION\]|\n\n|\Z)"
            r"(?:\[SUGGESTION\]\s*[:\-]?\s*(.*?)(?=\[SEVERITY\]|\n\n|\Z))?",
            re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            severity = match.group(1).upper()
            category = match.group(2).lower()
            line_str = match.group(3)
            desc = match.group(4).strip()
            sugg = (match.group(5) or "").strip()

            line_no = int(line_str) if line_str and line_str.isdigit() else None
            title = desc.split("\n")[0][:60]

            existing_findings.append(
                ReviewFinding(
                    title=title,
                    severity=severity if severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW") else "INFO",
                    category=category,
                    file_path=file_label,
                    line_number=line_no,
                    description=desc,
                    suggestion=sugg or "Review and refactor accordingly.",
                )
            )
