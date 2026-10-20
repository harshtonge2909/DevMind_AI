"""Streamlit Web Interface for DevMind AI - Intelligent Software Engineering Assistant."""

import os
import sys
from pathlib import Path

# Ensure src/ is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from devmind.config import DevMindConfig, load_config
from devmind.file_scanner import FileScanner
from devmind.indexer import ProjectIndexer
from devmind.models import CodeReviewReport, PatchProposal
from devmind.patch_manager import PatchManager
from devmind.retrieval import CodebaseRetriever
from devmind.utils import safe_read_file
from devmind.vector_store import VectorStoreManager
from devmind.workflows import WorkflowOrchestrator

# Configure Streamlit page
st.set_page_config(
    page_title="DevMind AI — Software Engineering Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-critical { background-color: #ffebee; color: #c62828; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-high { background-color: #fff3e0; color: #ef6c00; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #fffde7; color: #f57f17; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-info { background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .diff-container { background-color: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services():
    """Initialize core backend services."""
    config = load_config()
    vector_store = VectorStoreManager(config=config)
    orchestrator = WorkflowOrchestrator(config=config, vector_store=vector_store)
    patch_manager = PatchManager(config=config)
    return config, vector_store, orchestrator, patch_manager


config, vector_store, orchestrator, patch_manager = get_services()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_project" not in st.session_state:
    st.session_state.active_project = "project"
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = None
if "review_report" not in st.session_state:
    st.session_state.review_report = None


# -------------------------------------------------------------
# SIDEBAR: Project Configuration & Indexing
# -------------------------------------------------------------
with st.sidebar:
    st.title("🧠 DevMind AI")
    st.markdown("**Intelligent Engineering Assistant**")
    st.divider()

    # Project directory input
    project_dir_input = st.text_input(
        "📁 Project Directory",
        value=st.session_state.active_project,
        help="Path to the source code repository you wish to inspect.",
    )
    st.session_state.active_project = project_dir_input

    # API Key status
    api_ready = config.is_api_key_set()
    if api_ready:
        st.success("API Key: Configured", icon="✅")
    else:
        st.error("API Key: Not detected in .env", icon="⚠️")
        st.info("Set `Gemini_API_Key` or `GOOGLE_API_KEY` in `.env` to enable full AI workflows.")

    st.divider()

    # Indexing status
    manifest = orchestrator.indexer.load_manifest()
    st.subheader("📊 Index Status")
    if manifest and manifest.files:
        st.write(f"**Indexed Files:** {len(manifest.files)}")
        st.write(f"**Total Chunks:** {manifest.total_chunks}")
        st.write(f"**Model:** `{manifest.embedding_model}`")
        if manifest.indexed_at:
            dt = manifest.indexed_at[:19].replace("T", " ")
            st.caption(f"Last updated: {dt} UTC")
    else:
        st.warning("Project not yet indexed.")

    force_reindex = st.checkbox("Force full re-index", value=False)

    if st.button("⚡ Index / Update Codebase", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(cur, total, msg):
            progress_bar.progress(cur / 100.0)
            status_text.caption(msg)

        with st.spinner("Indexing codebase..."):
            result = orchestrator.indexer.index_project(
                project_path=st.session_state.active_project,
                force_reindex=force_reindex,
                progress_callback=on_progress,
            )

        if result.get("success"):
            st.success(
                f"Indexed {result.get('total_files', 0)} files ({result.get('total_chunks', 0)} chunks)!"
            )
            st.rerun()
        else:
            st.error(result.get("error", "Indexing failed."))

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# -------------------------------------------------------------
# MAIN VIEW: Workflows & Tabs
# -------------------------------------------------------------
st.markdown('<div class="main-header">DevMind AI Engineering Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Source-grounded RAG, AST symbol navigation, automated code reviews, safe patch application, and debugging.</div>', unsafe_allow_html=True)

tabs = st.tabs(
    [
        "💬 Code Q&A",
        "🔍 Explain Symbol",
        "🐛 Debug Assistant",
        "🛡️ Code Review",
        "🧪 Test Planner",
        "✏️ Patch & Refactor",
        "📁 Project Overview",
    ]
)

# -------------------------------------------------------------
# TAB 1: Code Q&A
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Grounded Codebase Q&A")

    if not manifest or not manifest.files:
        st.info("💡 Your project is not indexed yet. Click **'Index / Update Codebase'** in the sidebar to start asking questions.")

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Source Evidence & Citations"):
                    for s in msg["sources"]:
                        rel = s.get("rel_path", "source")
                        lines = s.get("lines", "")
                        st.markdown(f"**`{rel}`** ({lines})")
                        st.code(s.get("content", ""), language="python")

    # Chat Input
    user_query = st.chat_input("Ask a question about the codebase (e.g. 'How does add work?')...")
    if user_query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase and generating answer..."):
                workflow_res = orchestrator.ask(user_query)
                answer_text = workflow_res.answer

                st.markdown(answer_text)

                sources_data = []
                if workflow_res.sources:
                    with st.expander("📚 Source Evidence & Citations"):
                        for chunk in workflow_res.sources:
                            source_item = {
                                "rel_path": chunk.rel_path,
                                "lines": f"L{chunk.start_line}-L{chunk.end_line}",
                                "content": chunk.content,
                            }
                            sources_data.append(source_item)
                            st.markdown(f"**`{chunk.rel_path}`** (Lines {chunk.start_line}-{chunk.end_line})")
                            st.code(chunk.content, language=chunk.language or "python")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data,
                    }
                )

# -------------------------------------------------------------
# TAB 2: Explain Symbol
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Explain Function, Class, or File")
    col1, col2 = st.columns([3, 1])
    with col1:
        explain_target = st.text_input(
            "Symbol or File to Explain",
            placeholder="e.g., 'add', 'Calculator', or 'project/utils.py'",
        )
    with col2:
        st.write("")
        st.write("")
        explain_btn = st.button("Explain Target", type="primary", use_container_width=True)

    if explain_btn and explain_target:
        with st.spinner(f"Analyzing '{explain_target}'..."):
            result = orchestrator.explain(explain_target)
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Inspected Source"):
                    for s in result.sources:
                        st.markdown(f"**`{s.rel_path}`** (Lines {s.start_line}-{s.end_line})")
                        st.code(s.content, language="python")

# -------------------------------------------------------------
# TAB 3: Debug Assistant
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Python Traceback & Error Diagnostics")
    error_input = st.text_area(
        "Paste Error Message or Traceback",
        height=150,
        placeholder="Traceback (most recent call last):\n  File 'main.py', line 3, in <module>\n    result = add('a', 10)\nTypeError: can only concatenate str (not 'int') to str",
    )
    debug_code_context = st.text_area(
        "Additional Code Context (Optional)",
        height=100,
        placeholder="Paste any snippets or reproduction steps here...",
    )

    if st.button("Diagnose Error", type="primary"):
        if not error_input.strip():
            st.warning("Please paste an error message or traceback first.")
        else:
            with st.spinner("Diagnosing failure root cause..."):
                diag_result = orchestrator.debug(error_input, debug_code_context)
                st.markdown(diag_result.answer)

# -------------------------------------------------------------
# TAB 4: Code Review
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Static & AI-Assisted Code Review")

    indexed_files = list(manifest.files.keys()) if (manifest and manifest.files) else []
    file_to_review = st.selectbox(
        "Select File to Review",
        options=indexed_files or ["project/utils.py", "project/main.py"],
    )

    if st.button("Run Code Review", type="primary"):
        if not file_to_review:
            st.warning("Select or provide a file to review.")
        else:
            target_path = Path(st.session_state.active_project) / file_to_review
            if not target_path.exists():
                target_path = Path(file_to_review)

            with st.spinner(f"Reviewing {file_to_review}..."):
                report: CodeReviewReport = orchestrator.review(str(target_path))
                st.session_state.review_report = report

    if st.session_state.review_report:
        report = st.session_state.review_report
        st.divider()

        # Score card
        col_s1, col_s2 = st.columns([1, 3])
        with col_s1:
            score = report.score if report.score is not None else 100
            st.metric("Code Quality Score", f"{score}/100")
        with col_s2:
            st.markdown(f"**Summary:** {report.summary[:300]}...")

        st.subheader("Findings")
        if not report.findings:
            st.success("No critical or high severity issues identified in this file!")
        else:
            badge_map = {
                "CRITICAL": "badge-critical",
                "HIGH": "badge-high",
                "MEDIUM": "badge-medium",
                "LOW": "badge-low",
                "INFO": "badge-info",
            }
            for f in report.findings:
                badge_class = badge_map.get(f.severity.upper(), "badge-info")
                line_info = f"Line {f.line_number}" if f.line_number else "Global"
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 12px; margin-bottom: 10px; border-radius: 6px;">
                        <span class="{badge_class}">{f.severity}</span>
                        <strong>{f.title}</strong> — <small>{line_info} | {f.category}</small>
                        <p style="margin: 6px 0;">{f.description}</p>
                        <p style="margin: 0; color: #1565C0;"><strong>Suggestion:</strong> {f.suggestion}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if report.recommendations:
            with st.expander("Recommendations"):
                for rec in report.recommendations:
                    st.markdown(f"- {rec}")

# -------------------------------------------------------------
# TAB 5: Test Planner
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("Automated Test Suite Planner")
    test_target = st.text_input(
        "Target Component or File for Testing",
        placeholder="e.g. 'add' or 'project/utils.py'",
    )

    if st.button("Generate Test Plan & Pytest Suite", type="primary"):
        if not test_target.strip():
            st.warning("Please specify a target function, class, or file.")
        else:
            with st.spinner("Designing test strategy and test cases..."):
                test_res = orchestrator.plan_tests(test_target)
                st.markdown(test_res.answer)

# -------------------------------------------------------------
# TAB 6: Patch & Refactor
# -------------------------------------------------------------
with tabs[5]:
    st.subheader("Safe Code Refactoring & Patch Proposals")
    st.caption("Inspect proposed diffs, review safeguards, and confirm before applying changes.")

    patch_file = st.text_input(
        "File to Modify",
        value="project/utils.py",
    )
    patch_instructions = st.text_area(
        "Refactoring or Bug Fix Instructions",
        placeholder="e.g., 'Add type annotations, a docstring, and input validation to add()'",
    )

    if st.button("Propose Code Change", type="primary"):
        if not patch_file or not patch_instructions:
            st.warning("Please specify both target file and instructions.")
        else:
            with st.spinner("Generating proposed patch and diff..."):
                target_path = Path(st.session_state.active_project) / patch_file
                if not target_path.exists():
                    target_path = Path(patch_file)

                proposal = orchestrator.propose_patch(str(target_path), patch_instructions)
                st.session_state.current_proposal = proposal

    if st.session_state.current_proposal:
        prop: PatchProposal = st.session_state.current_proposal
        st.divider()
        st.markdown(f"### Proposed Patch for `{prop.rel_path}`")
        st.markdown(prop.explanation)

        st.subheader("Unified Diff Preview")
        if prop.unified_diff:
            st.code(prop.unified_diff, language="diff")
        else:
            st.info("No text changes detected between original and proposed code.")

        st.markdown("#### Regression Risks & Verification")
        for risk in prop.regression_risks:
            st.markdown(f"- ⚠️ {risk}")

        st.divider()
        st.markdown("### Safety Controls & Application")

        confirm_checkbox = st.checkbox(
            "I have reviewed the unified diff above and explicitly confirm applying this change.",
            value=False,
        )

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            if st.button(
                "Apply Patch to File",
                type="primary",
                disabled=not confirm_checkbox,
                use_container_width=True,
            ):
                apply_res = patch_manager.apply_patch(
                    target_file=prop.target_file,
                    proposed_code=prop.proposed_code,
                    confirmed=confirm_checkbox,
                )
                if apply_res.get("success"):
                    st.success(apply_res.get("message"))
                    st.session_state.current_proposal = None
                else:
                    st.error(apply_res.get("error", "Failed to apply patch."))

        with col_a2:
            if st.button("Rollback Last Patch", use_container_width=True):
                rb_res = patch_manager.rollback()
                if rb_res.get("success"):
                    st.success(rb_res.get("message"))
                else:
                    st.error(rb_res.get("error", "Rollback failed."))

# -------------------------------------------------------------
# TAB 7: Project Overview
# -------------------------------------------------------------
with tabs[6]:
    st.subheader("Project Structure & Architecture Analysis")

    if st.button("Generate Architectural Overview", type="primary"):
        with st.spinner("Analyzing project architecture and dependencies..."):
            arch_res = orchestrator.analyze_project(st.session_state.active_project)
            st.markdown(arch_res.answer)

    st.divider()
    st.subheader("Discovered Files & Symbols")
    if manifest and manifest.files:
        for rel_file, file_info in manifest.files.items():
            with st.expander(f"📄 {rel_file} ({file_info.get('chunk_count', 0)} chunks)"):
                st.write(f"Language: `{file_info.get('language', 'unknown')}`")
                st.write(f"Hash: `{file_info.get('file_hash', '')[:16]}...`")
                st.write(f"Size: {file_info.get('size', 0)} bytes")
