"""LangChain workflow orchestration for DevMind AI."""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from devmind.code_parser import CodeParser
from devmind.code_review import CodeReviewer
from devmind.config import DevMindConfig, load_config
from devmind.indexer import ProjectIndexer
from devmind.models import (
    CodeReviewReport,
    PatchProposal,
    RetrievalResult,
    SourceChunk,
    WorkflowResult,
)
from devmind.patch_manager import PatchManager
from devmind.prompts import (
    ASK_SYSTEM_PROMPT,
    ASK_USER_PROMPT,
    DEBUG_SYSTEM_PROMPT,
    DEBUG_USER_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
    EXPLAIN_USER_PROMPT,
    PATCH_PROPOSAL_SYSTEM_PROMPT,
    PATCH_PROPOSAL_USER_PROMPT,
    PROJECT_ANALYSIS_SYSTEM_PROMPT,
    PROJECT_ANALYSIS_USER_PROMPT,
    TEST_PLAN_SYSTEM_PROMPT,
    TEST_PLAN_USER_PROMPT,
)
from devmind.retrieval import CodebaseRetriever
from devmind.utils import extract_text_content, safe_read_file
from devmind.vector_store import VectorStoreManager


class WorkflowOrchestrator:
    """Coordinates DevMind AI multi-step workflows with resilient error handling."""

    def __init__(
        self,
        config: Optional[DevMindConfig] = None,
        llm: Optional[BaseChatModel] = None,
        vector_store: Optional[VectorStoreManager] = None,
        code_parser: Optional[CodeParser] = None,
    ):
        self.config = config or load_config()
        self._llm = llm
        self.vector_store = vector_store or VectorStoreManager(config=self.config)
        self.code_parser = code_parser or CodeParser()
        self.retriever = CodebaseRetriever(
            config=self.config,
            vector_store=self.vector_store,
            code_parser=self.code_parser,
        )
        self.patch_manager = PatchManager(config=self.config)
        self.indexer = ProjectIndexer(
            config=self.config,
            vector_store=self.vector_store,
            code_parser=self.code_parser,
        )

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize Google Gemini chat model."""
        if self._llm is None:
            api_key = self.config.get_api_key()
            if not api_key:
                raise ValueError(
                    "Google API Key is not configured. Please set 'Gemini_API_Key' "
                    "or 'GOOGLE_API_KEY' in your .env file."
                )
            init_kwargs = {
                "model": self.config.model_name,
                "google_api_key": api_key,
            }
            # Only set temperature if not using models with fixed sampling defaults
            if self.config.temperature is not None and "gemini-3.6" not in self.config.model_name:
                init_kwargs["temperature"] = self.config.temperature

            self._llm = ChatGoogleGenerativeAI(**init_kwargs)
        return self._llm

    def _invoke_with_retry(self, messages: List[Any], max_retries: int = 3) -> str:
        """Invoke Gemini LLM with bounded exponential backoff for transient errors."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return extract_text_content(response.content)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Retry on 503, rate limits, or transient connection errors
                is_transient = any(
                    code in err_str
                    for code in ["503", "unavailable", "resourceexhausted", "timeout", "reset"]
                )
                if is_transient and attempt < max_retries:
                    backoff = min(2**attempt, 8)
                    time.sleep(backoff)
                    continue
                break

        raise RuntimeError(f"Gemini API invocation failed: {str(last_error)}")

    # -------------------------------------------------------------
    # 1. Ask Workflow
    # -------------------------------------------------------------
    def ask(self, question: str, top_k: Optional[int] = None) -> WorkflowResult:
        """Answer questions using retrieved codebase evidence."""
        if not question or not question.strip():
            return WorkflowResult(
                workflow_type="ask",
                success=False,
                answer="Question cannot be empty.",
                error="Empty question",
            )

        try:
            results = self.retriever.hybrid_retrieve(question, top_k=top_k)
            evidence = self.retriever.format_evidence_prompt(results)

            messages = [
                SystemMessage(content=ASK_SYSTEM_PROMPT),
                HumanMessage(content=ASK_USER_PROMPT.format(question=question, context=evidence)),
            ]
            answer = self._invoke_with_retry(messages)

            return WorkflowResult(
                workflow_type="ask",
                success=True,
                answer=answer,
                sources=[r.chunk for r in results],
                metadata={"top_k": top_k or self.config.default_top_k},
            )
        except Exception as e:
            return WorkflowResult(
                workflow_type="ask",
                success=False,
                answer=f"Error executing Ask workflow: {str(e)}",
                error=str(e),
            )

    # -------------------------------------------------------------
    # 2. Explain Workflow
    # -------------------------------------------------------------
    def explain(self, target: str) -> WorkflowResult:
        """Explain a function, class, module, or file."""
        if not target or not target.strip():
            return WorkflowResult(
                workflow_type="explain",
                success=False,
                answer="Target symbol or file cannot be empty.",
                error="Empty target",
            )

        try:
            target_clean = target.strip()
            # 1. Check if target is a file
            content = safe_read_file(target_clean)
            context = ""
            sources: List[SourceChunk] = []

            if content:
                context = f"=== File: {target_clean} ===\n{content[:6000]}"
                sources.append(
                    SourceChunk(
                        content=content[:1000],
                        file_path=target_clean,
                        rel_path=target_clean,
                        start_line=1,
                        end_line=len(content.splitlines()),
                        chunk_index=0,
                        chunk_id=f"file::{target_clean}",
                    )
                )
            else:
                # 2. Check if target is a symbol
                results = self.retriever.retrieve_by_symbol(target_clean)
                if not results:
                    results = self.retriever.retrieve(query=f"function or class {target_clean}", top_k=3)

                context = self.retriever.format_evidence_prompt(results)
                sources = [r.chunk for r in results]

            messages = [
                SystemMessage(content=EXPLAIN_SYSTEM_PROMPT),
                HumanMessage(content=EXPLAIN_USER_PROMPT.format(target=target_clean, context=context)),
            ]
            answer = self._invoke_with_retry(messages)

            return WorkflowResult(
                workflow_type="explain",
                success=True,
                answer=answer,
                sources=sources,
                metadata={"target": target_clean},
            )
        except Exception as e:
            return WorkflowResult(
                workflow_type="explain",
                success=False,
                answer=f"Error executing Explain workflow: {str(e)}",
                error=str(e),
            )

    # -------------------------------------------------------------
    # 3. Debug Workflow
    # -------------------------------------------------------------
    def debug(
        self, error_trace: str, context_code: Optional[str] = None
    ) -> WorkflowResult:
        """Analyze an error traceback and diagnose root cause with fixes."""
        if not error_trace or not error_trace.strip():
            return WorkflowResult(
                workflow_type="debug",
                success=False,
                answer="Error traceback or message cannot be empty.",
                error="Empty error trace",
            )

        try:
            context = context_code or ""
            sources: List[SourceChunk] = []

            # Extract any file paths or function names mentioned in the traceback
            words = re.findall(r'File "([^"]+)", line (\d+)', error_trace)
            for file_path, lineno in words:
                line_int = int(lineno)
                file_text = safe_read_file(file_path)
                if file_text:
                    lines = file_text.splitlines()
                    start = max(0, line_int - 10)
                    end = min(len(lines), line_int + 10)
                    snippet = "\n".join(lines[start:end])
                    context += f"\n--- Referenced File: {file_path} (Lines {start+1}-{end}) ---\n{snippet}\n"
                    sources.append(
                        SourceChunk(
                            content=snippet,
                            file_path=file_path,
                            rel_path=file_path,
                            start_line=start + 1,
                            end_line=end,
                            chunk_index=0,
                            chunk_id=f"debug::{file_path}::L{line_int}",
                        )
                    )

            if not context:
                # Fallback to vector search for terms in the traceback
                first_line = error_trace.strip().splitlines()[-1]
                retrieval = self.retriever.retrieve(query=first_line, top_k=2)
                context = self.retriever.format_evidence_prompt(retrieval)
                sources = [r.chunk for r in retrieval]

            messages = [
                SystemMessage(content=DEBUG_SYSTEM_PROMPT),
                HumanMessage(
                    content=DEBUG_USER_PROMPT.format(
                        error_trace=error_trace, context=context or "No relevant source code found."
                    )
                ),
            ]
            answer = self._invoke_with_retry(messages)

            return WorkflowResult(
                workflow_type="debug",
                success=True,
                answer=answer,
                sources=sources,
            )
        except Exception as e:
            return WorkflowResult(
                workflow_type="debug",
                success=False,
                answer=f"Error executing Debug workflow: {str(e)}",
                error=str(e),
            )

    # -------------------------------------------------------------
    # 4. Review Workflow
    # -------------------------------------------------------------
    def review(self, target_file: str) -> CodeReviewReport:
        """Run code review on a file with security and quality findings."""
        reviewer = CodeReviewer(config=self.config, llm=self.llm if self.config.is_api_key_set() else None)
        return reviewer.review_file(target_file)

    # -------------------------------------------------------------
    # 5. Test Planning Workflow
    # -------------------------------------------------------------
    def plan_tests(self, target: str) -> WorkflowResult:
        """Recommend automated testing strategies and runnable pytest tests."""
        if not target or not target.strip():
            return WorkflowResult(
                workflow_type="test_plan",
                success=False,
                answer="Target cannot be empty.",
                error="Empty target",
            )

        try:
            target_clean = target.strip()
            content = safe_read_file(target_clean)
            context = ""
            sources: List[SourceChunk] = []

            if content:
                context = f"=== File: {target_clean} ===\n{content[:6000]}"
                sources.append(
                    SourceChunk(
                        content=content[:1000],
                        file_path=target_clean,
                        rel_path=target_clean,
                        start_line=1,
                        end_line=len(content.splitlines()),
                        chunk_index=0,
                        chunk_id=f"test_plan::{target_clean}",
                    )
                )
            else:
                results = self.retriever.retrieve_by_symbol(target_clean)
                if not results:
                    results = self.retriever.retrieve(query=target_clean, top_k=3)
                context = self.retriever.format_evidence_prompt(results)
                sources = [r.chunk for r in results]

            messages = [
                SystemMessage(content=TEST_PLAN_SYSTEM_PROMPT),
                HumanMessage(content=TEST_PLAN_USER_PROMPT.format(target=target_clean, context=context)),
            ]
            answer = self._invoke_with_retry(messages)

            return WorkflowResult(
                workflow_type="test_plan",
                success=True,
                answer=answer,
                sources=sources,
                metadata={"target": target_clean},
            )
        except Exception as e:
            return WorkflowResult(
                workflow_type="test_plan",
                success=False,
                answer=f"Error executing Test Planning workflow: {str(e)}",
                error=str(e),
            )

    # -------------------------------------------------------------
    # 6. Project Analysis Workflow
    # -------------------------------------------------------------
    def analyze_project(self, project_path: Optional[str] = None) -> WorkflowResult:
        """Produce structured overview of architecture, dependencies, and modules."""
        target_path = project_path or "project"
        try:
            manifest = self.indexer.load_manifest()
            summary = self.code_parser.get_project_summary()

            summary_text = (
                f"- Indexed Files: {len(summary.get('files', []))}\n"
                f"- Total Lines of Code: {summary.get('total_lines', 0)}\n"
                f"- Classes: {', '.join(summary.get('classes', [])) or 'None'}\n"
                f"- Functions: {', '.join(summary.get('functions', [])) or 'None'}\n"
                f"- Module Dependencies: {summary.get('module_dependencies', {})}"
            )

            messages = [
                SystemMessage(content=PROJECT_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=PROJECT_ANALYSIS_USER_PROMPT.format(
                        project_path=target_path,
                        total_files=summary.get("total_files", 0),
                        total_lines=summary.get("total_lines", 0),
                        summary_data=summary_text,
                    )
                ),
            ]
            answer = self._invoke_with_retry(messages)

            return WorkflowResult(
                workflow_type="project_analysis",
                success=True,
                answer=answer,
                metadata=summary,
            )
        except Exception as e:
            return WorkflowResult(
                workflow_type="project_analysis",
                success=False,
                answer=f"Error executing Project Analysis: {str(e)}",
                error=str(e),
            )

    # -------------------------------------------------------------
    # 7. Code Change Proposal Workflow
    # -------------------------------------------------------------
    def propose_patch(
        self, target_file: str, instructions: str
    ) -> PatchProposal:
        """Generate a proposed patch and unified diff without applying it automatically."""
        orig_code = safe_read_file(target_file)
        if orig_code is None:
            return self.patch_manager.create_proposal(
                target_file=target_file,
                rel_path=Path(target_file).name,
                proposed_code="",
                explanation=f"Cannot read target file: {target_file}",
            )

        rel_label = Path(target_file).name
        # Context from retrieval
        results = self.retriever.retrieve(query=instructions, top_k=2)
        context = self.retriever.format_evidence_prompt(results)

        messages = [
            SystemMessage(content=PATCH_PROPOSAL_SYSTEM_PROMPT),
            HumanMessage(
                content=PATCH_PROPOSAL_USER_PROMPT.format(
                    file_path=rel_label,
                    original_code=orig_code,
                    instructions=instructions,
                    context=context,
                )
            ),
        ]

        llm_output = self._invoke_with_retry(messages)

        # Extract full replacement code block from markdown
        code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if code_match:
            proposed_code = code_match.group(1)
        else:
            proposed_code = orig_code

        explanation = llm_output
        proposal = self.patch_manager.create_proposal(
            target_file=target_file,
            rel_path=rel_label,
            proposed_code=proposed_code,
            explanation=explanation,
            regression_risks=[
                "Check callers of modified functions for interface compatibility.",
                "Ensure edge cases and input validation remain consistent.",
            ],
            suggested_tests=[
                f"pytest tests/ -k {Path(target_file).stem}",
                "Verify return values with sample inputs.",
            ],
        )

        return proposal
