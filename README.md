# DevMind AI — Intelligent Software Engineering Assistant

DevMind AI is a full-featured, AI-powered software engineering assistant built with **Python 3.12**, **LangChain**, **Google Gemini**, and **ChromaDB**. It provides source-grounded code question answering, deep AST symbol navigation, automated code reviews, python runtime diagnostics, test strategy generation, and safe code refactoring with unified diffs and rollbacks.

---

## 🌟 Key Features

1. **Reliable Codebase RAG Pipeline (Phase 1)**
   - Recursive source file discovery supporting Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more.
   - Comprehensive exclusion rules for `.venv`, `node_modules`, `__pycache__`, build directories, and binary files.
   - Strict secret sanitization preventing ingestion of `.env` files, private keys (`id_rsa`, `.pem`), and tokens.
   - Line-aware chunking preserving exact `start_line` and `end_line` metadata for traceable source citations.
   - Persistent ChromaDB vector storage with Google `gemini-embedding-001` (3072-dimensional vectors).
   - High-performance incremental indexing: detects added, modified, and deleted files using SHA-256 hashes to prevent stale or duplicate vectors.

2. **Codebase Understanding & Intelligent Search (Phase 2)**
   - Python AST (`ast` module) symbol extractor for functions, classes, methods, docstrings, parameters, and call-sites.
   - Hybrid retrieval combining vector similarity search with exact AST symbol grounding.
   - Cross-module relationship and dependency discovery.

3. **Debugging & Code Review Assistant (Phase 3)**
   - Traceback and runtime exception diagnostics mapping stack traces back to source lines.
   - Dual-layer code review: static heuristic analysis (security vulnerabilities, bare exceptions, mutable default arguments, hardcoded credentials) + deep LLM architectural review with severity badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`).
   - Safe patching engine: unified diff previews, required explicit confirmation, pre-patch AST syntax validation, and automated safety backups with one-click rollback.

4. **Multi-Step LangChain Workflows (Phase 4)**
   - 7 Modular workflows: **Ask**, **Explain**, **Debug**, **Review**, **Test Planning**, **Project Analysis**, and **Code Change Proposal**.
   - Resilient execution with bounded exponential backoff for transient API errors (e.g., HTTP 503 / rate limits).

5. **Polished Web Interface (Phase 5)**
   - Built with Streamlit for a fast, responsive UI on Windows.
   - Project directory selector, real-time indexing status card, and progress bars.
   - Multi-turn conversational chat with expandable source evidence cards and line citations.
   - Dedicated workspace tabs for Q&A, Symbol Explanation, Traceback Debugger, Code Review Findings, Test Planning, and Diff-based Patching.

6. **Automated Test Suite (Phase 6)**
   - 31 unit and integration tests using `pytest` verifying file scanning, secret exclusions, AST parsing, chunking, retrieval formatting, safeguards, and Streamlit UI startup.
   - Complete offline test execution with mock fixtures.

---

## 🏗️ Architecture & Project Structure

```text
DevMind AI/
├── main.py                     # Interactive CLI and direct command runner
├── app.py                      # Streamlit web application
├── pyproject.toml              # Dependencies & package configuration
├── uv.lock                     # Locked dependency tree
├── .env                        # Local API credentials (NEVER committed)
├── .env.example                # Example template without secrets
├── .gitignore                  # Git ignore rules protecting secrets & vector db
├── README.md                   # Full system documentation
├── src/
│   └── devmind/
│       ├── __init__.py         # Package entrypoint
│       ├── config.py           # Configuration, defaults, and safe env loader
│       ├── models.py           # Typed dataclasses & data schemas
│       ├── file_scanner.py     # Recursive scanner, .gitignore, secret filters
│       ├── code_parser.py      # Python AST parser, symbols, calls & relationships
│       ├── indexer.py          # Line-aware chunking & incremental indexing
│       ├── vector_store.py     # ChromaDB persistence & vector management
│       ├── retrieval.py        # Hybrid symbol & semantic retriever
│       ├── prompts.py          # Structured prompt templates
│       ├── workflows.py        # LangChain workflow orchestrator
│       ├── code_review.py      # Static security heuristics & LLM review
│       ├── patch_manager.py    # Unified diffs, backups, safeguards & rollback
│       └── utils.py            # Text extraction, hashing, Windows path helpers
├── tests/
│   ├── conftest.py             # Shared pytest fixtures & offline mocks
│   ├── test_app.py             # Streamlit application UI test
│   ├── test_file_scanner.py    # Discovery, gitignore & secret exclusion tests
│   ├── test_code_parser.py     # AST symbol & relationship tests
│   ├── test_indexer.py         # Incremental indexing & chunking tests
│   ├── test_retrieval.py       # Retrieval & citation formatting tests
│   ├── test_patch_manager.py   # Diff, safety confirmation & rollback tests
│   └── test_workflows.py       # Workflow validation & error handling tests
├── project/                    # Target project directory to analyze
├── data/                       # Local index manifests (git-ignored)
└── chroma_db/                  # Persistent vector database (git-ignored)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **uv** (recommended for ultra-fast dependency management) or standard Python `pip`
- Windows 10/11, macOS, or Linux

### 1. Installation

Clone or open the project folder in your terminal:

```powershell
cd "D:\DevMind AI"
```

Sync dependencies using `uv`:

```powershell
uv sync
```

Alternatively, install dependencies using `uv pip`:

```powershell
uv pip install -r pyproject.toml
```

### 2. Environment Configuration

DevMind AI requires a Google Gemini API key.
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Copy the `.env.example` file to `.env`:

```powershell
Copy-Item .env.example .env
```

3. Open `.env` and set your key:

```env
Gemini_API_Key=your_actual_gemini_api_key_here
```

*(Note: DevMind AI also accepts `GOOGLE_API_KEY` or `GEMINI_API_KEY`).*

---

## 💻 Usage

### A. Streamlit Web Interface

Launch the interactive web application in your browser:

```powershell
uv run streamlit run app.py
```

Or run via the CLI shortcut:

```powershell
uv run python main.py web
```

The application will open at `http://localhost:8501`.

**Features in the Web UI:**
- **Sidebar**: Select project directory, inspect total indexed chunks/files, and click **"Index / Update Codebase"**.
- **💬 Code Q&A**: Ask natural-language questions and inspect expandable source evidence with line citations.
- **🔍 Explain Symbol**: Enter any function or class name (e.g., `add`) for architectural analysis.
- **🐛 Debug Assistant**: Paste Python error tracebacks for root-cause diagnosis.
- **🛡️ Code Review**: Run static and deep audits with severity ratings and actionable refactoring suggestions.
- **🧪 Test Planner**: Generate comprehensive pytest test suites with boundary cases and test matrices.
- **✏️ Patch & Refactor**: Generate unified diffs, review regression risks, and click "Apply Patch Safely" with automated backups.

---

### B. Interactive CLI

Start the interactive terminal REPL:

```powershell
uv run python main.py
```

Available interactive commands:
- `/index [path]` — Index or incrementally update repository vectors (e.g. `/index project`).
- `/ask <question>` — Ask questions grounded in code (e.g. `/ask How does add work?`).
- `/explain <symbol>` — Deeply explain a function, class, or module (e.g. `/explain add`).
- `/debug` — Enter multiline error traceback diagnosis mode.
- `/review <file>` — Audit a file for security and style issues (e.g. `/review project/utils.py`).
- `/test <target>` — Generate automated pytest suites (e.g. `/test add`).
- `/patch <file>` — Generate and preview a unified diff, then apply safely.
- `/rollback` — Restore the previous file state from backup.
- `/summary` — Display project architecture, entry points, and dependencies.
- `/files` — List currently indexed files, chunk counts, and languages.
- `/help` — Display commands help.
- `/exit` — Quit CLI.

---

### C. Direct CLI Commands

You can run individual workflows directly from your shell:

```powershell
# Index a project
uv run python main.py index project/

# Ask a question
uv run python main.py ask "What does add do?"

# Explain a symbol
uv run python main.py explain add

# Review a source file
uv run python main.py review project/utils.py

# Plan tests
uv run python main.py test project/utils.py
```

---

## 🧪 Running Automated Tests

DevMind AI includes a comprehensive unit and integration test suite. All tests execute completely offline by mocking external API calls:

```powershell
uv run pytest
```

To run with verbose output:

```powershell
uv run pytest -v
```

---

## 🔒 Security & Safe Operations

- **API Key Protection**: API keys are loaded server-side through `devmind.config` and are never exposed in UI responses, logs, or git commits.
- **Secret Filtering**: The file scanner automatically prevents indexing of `.env`, `.pem`, `.key`, `id_rsa`, certificates, and files containing credential headers.
- **Safe Code Modification**: Patches require explicit user confirmation, run pre-modification Python AST syntax checks, and create timestamped backups in `.devmind_backups/` before any file write.
- **Isolated Workflows**: DevMind AI never executes arbitrary shell commands or untrusted AI code.
