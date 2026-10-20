"""DevMind AI - Interactive Command-Line Interface and Workflow Runner."""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from devmind.config import DevMindConfig, load_config
from devmind.models import CodeReviewReport, PatchProposal
from devmind.patch_manager import PatchManager
from devmind.vector_store import VectorStoreManager
from devmind.workflows import WorkflowOrchestrator

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console(legacy_windows=False)


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                 🧠  DevMind AI v0.1.0                   ║
    ║        Intelligent Software Engineering Assistant        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(f"[bold cyan]{banner}[/bold cyan]")


def print_help():
    table = Table(title="DevMind AI - Available Interactive Commands", show_header=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Arguments", style="yellow")
    table.add_column("Description", style="white")

    table.add_row("/index", "[path] [--force]", "Index or incrementally update repository vectors")
    table.add_row("/ask", "<question>", "Ask question grounded strictly in codebase context")
    table.add_row("/explain", "<symbol/file>", "Explain a function, class, or module")
    table.add_row("/debug", "", "Enter interactive error / traceback diagnosis mode")
    table.add_row("/review", "<file>", "Run security and code quality review on a file")
    table.add_row("/test", "<target>", "Generate automated test plan and pytest test suite")
    table.add_row("/patch", "<file>", "Propose code changes, view unified diff, and apply safely")
    table.add_row("/rollback", "", "Rollback last applied patch from safety backup")
    table.add_row("/summary", "", "Generate architectural analysis and module dependencies")
    table.add_row("/files", "", "List currently indexed files and symbols")
    table.add_row("/help", "", "Show this commands guide")
    table.add_row("/exit", "", "Exit DevMind AI CLI")

    console.print(table)


def run_interactive(orchestrator: WorkflowOrchestrator, patch_manager: PatchManager):
    print_banner()
    console.print("[dim]Type [cyan]/help[/cyan] to view available commands or [cyan]/exit[/cyan] to quit.[/dim]\n")

    current_project = "project"

    while True:
        try:
            user_input = console.input("[bold green]devmind>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]DevMind AI shutting down... Goodbye![/yellow]")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "exit", "quit", ":q"):
            console.print("[yellow]DevMind AI shutting down... Goodbye![/yellow]")
            break

        elif cmd in ("/help", "help"):
            print_help()

        elif cmd in ("/index", "index"):
            target_path = arg.strip() or current_project
            force = False
            if "--force" in target_path:
                force = True
                target_path = target_path.replace("--force", "").strip() or current_project

            with console.status(f"[bold cyan]Indexing codebase at '{target_path}'...[/bold cyan]"):
                res = orchestrator.indexer.index_project(
                    project_path=target_path,
                    force_reindex=force,
                )

            if res.get("success"):
                console.print(
                    f"[green]✔ Indexing completed successfully![/green] "
                    f"Files: {res.get('total_files', 0)} | Chunks: {res.get('total_chunks', 0)} "
                    f"(Added: {res.get('added_files', 0)}, Modified: {res.get('modified_files', 0)}, Deleted: {res.get('deleted_files', 0)})"
                )
                current_project = target_path
            else:
                console.print(f"[bold red]Indexing failed:[/bold red] {res.get('error')}")

        elif cmd in ("/ask", "ask"):
            question = arg.strip()
            if not question:
                console.print("[yellow]Usage: /ask <your question about the code>[/yellow]")
                continue

            with console.status("[bold cyan]Retrieving context & generating answer...[/bold cyan]"):
                result = orchestrator.ask(question)

            if result.success:
                console.print(Panel(Markdown(result.answer), title="🧠 DevMind Answer", border_style="cyan"))
                if result.sources:
                    console.print("\n[dim bold]📚 Source Citations & Evidence:[/dim bold]")
                    for s in result.sources:
                        console.print(f"  • [cyan]{s.rel_path}[/cyan] (Lines {s.start_line}-{s.end_line})")
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")

        elif cmd in ("/explain", "explain"):
            target = arg.strip()
            if not target:
                console.print("[yellow]Usage: /explain <function, class, or file>[/yellow]")
                continue

            with console.status(f"[bold cyan]Explaining '{target}'...[/bold cyan]"):
                result = orchestrator.explain(target)

            if result.success:
                console.print(Panel(Markdown(result.answer), title=f"🔍 Explanation: {target}", border_style="blue"))
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")

        elif cmd in ("/debug", "debug"):
            console.print("[cyan]Paste your error / traceback below. Press Enter on an empty line to submit:[/cyan]")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            error_trace = "\n".join(lines).strip()

            if not error_trace:
                console.print("[yellow]No traceback provided. Aborted.[/yellow]")
                continue

            with console.status("[bold cyan]Diagnosing root cause and preparing fix...[/bold cyan]"):
                result = orchestrator.debug(error_trace)

            if result.success:
                console.print(Panel(Markdown(result.answer), title="🐛 Debug Diagnostics & Fix", border_style="red"))
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")

        elif cmd in ("/review", "review"):
            target_file = arg.strip()
            if not target_file:
                console.print("[yellow]Usage: /review <file_path>[/yellow]")
                continue

            with console.status(f"[bold cyan]Reviewing '{target_file}'...[/bold cyan]"):
                report: CodeReviewReport = orchestrator.review(target_file)

            console.print(Panel(Markdown(report.summary), title=f"🛡️ Code Review: {target_file}", border_style="magenta"))

            if report.findings:
                table = Table(title=f"Findings (Quality Score: {report.score}/100)", show_header=True)
                table.add_column("Severity", style="bold")
                table.add_column("Line", style="dim")
                table.add_column("Title", style="white")
                table.add_column("Suggestion", style="cyan")

                severity_colors = {
                    "CRITICAL": "bold red",
                    "HIGH": "red",
                    "MEDIUM": "yellow",
                    "LOW": "green",
                    "INFO": "blue",
                }

                for f in report.findings:
                    col = severity_colors.get(f.severity.upper(), "white")
                    line_str = str(f.line_number) if f.line_number else "-"
                    table.add_row(f"[{col}]{f.severity}[/{col}]", line_str, f.title, f.suggestion)

                console.print(table)
            else:
                console.print("[green]✔ No issues detected by static analysis![/green]")

        elif cmd in ("/test", "test"):
            target = arg.strip()
            if not target:
                console.print("[yellow]Usage: /test <function, class, or file>[/yellow]")
                continue

            with console.status(f"[bold cyan]Generating test strategy for '{target}'...[/bold cyan]"):
                result = orchestrator.plan_tests(target)

            if result.success:
                console.print(Panel(Markdown(result.answer), title=f"🧪 Test Plan: {target}", border_style="green"))
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")

        elif cmd in ("/patch", "patch"):
            target_file = arg.strip()
            if not target_file:
                console.print("[yellow]Usage: /patch <file_path>[/yellow]")
                continue

            instructions = console.input("[bold yellow]Instructions for change: [/bold yellow]").strip()
            if not instructions:
                console.print("[yellow]Instructions cannot be empty. Aborted.[/yellow]")
                continue

            with console.status("[bold cyan]Generating patch proposal and unified diff...[/bold cyan]"):
                proposal: PatchProposal = orchestrator.propose_patch(target_file, instructions)

            console.print(Panel(Markdown(proposal.explanation), title="Patch Explanation", border_style="yellow"))

            if proposal.unified_diff:
                console.print("\n[bold]Proposed Unified Diff:[/bold]")
                syntax = Syntax(proposal.unified_diff, "diff", theme="monokai", line_numbers=True)
                console.print(syntax)

                confirm = console.input("\n[bold red]Apply this patch to the file? (yes/no): [/bold red]").strip().lower()
                if confirm in ("yes", "y"):
                    apply_res = patch_manager.apply_patch(
                        target_file=proposal.target_file,
                        proposed_code=proposal.proposed_code,
                        confirmed=True,
                    )
                    if apply_res.get("success"):
                        console.print(f"[green]✔ {apply_res.get('message')}[/green]")
                    else:
                        console.print(f"[bold red]Error applying patch:[/bold red] {apply_res.get('error')}")
                else:
                    console.print("[yellow]Patch application cancelled by user.[/yellow]")
            else:
                console.print("[yellow]No code changes generated for this request.[/yellow]")

        elif cmd in ("/rollback", "rollback"):
            with console.status("[bold cyan]Rolling back last applied patch...[/bold cyan]"):
                rb_res = patch_manager.rollback()

            if rb_res.get("success"):
                console.print(f"[green]✔ {rb_res.get('message')}[/green]")
            else:
                console.print(f"[bold red]Rollback error:[/bold red] {rb_res.get('error')}")

        elif cmd in ("/summary", "summary"):
            with console.status("[bold cyan]Generating project architecture summary...[/bold cyan]"):
                result = orchestrator.analyze_project(current_project)

            if result.success:
                console.print(Panel(Markdown(result.answer), title="🏗️ Project Architecture Overview", border_style="magenta"))
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")

        elif cmd in ("/files", "files"):
            manifest = orchestrator.indexer.load_manifest()
            if manifest and manifest.files:
                table = Table(title=f"Indexed Files ({len(manifest.files)})", show_header=True)
                table.add_column("File Path", style="cyan")
                table.add_column("Language", style="yellow")
                table.add_column("Chunks", style="green")
                table.add_column("Size", style="dim")

                for rel_path, meta in manifest.files.items():
                    table.add_row(
                        rel_path,
                        meta.get("language", "unknown"),
                        str(meta.get("chunk_count", 0)),
                        f"{meta.get('size', 0)} B",
                    )
                console.print(table)
            else:
                console.print("[yellow]No files indexed yet. Run /index to index the project.[/yellow]")

        else:
            # Default fallback: Treat input directly as question
            with console.status("[bold cyan]Retrieving context & answering question...[/bold cyan]"):
                result = orchestrator.ask(user_input)

            if result.success:
                console.print(Panel(Markdown(result.answer), title="🧠 DevMind Answer", border_style="cyan"))
                if result.sources:
                    console.print("\n[dim bold]📚 Source Citations & Evidence:[/dim bold]")
                    for s in result.sources:
                        console.print(f"  • [cyan]{s.rel_path}[/cyan] (Lines {s.start_line}-{s.end_line})")
            else:
                console.print(f"[bold red]Error:[/bold red] {result.error or result.answer}")


def main():
    parser = argparse.ArgumentParser(description="DevMind AI - Software Engineering Assistant")
    parser.add_argument("command", nargs="?", default="interactive", help="Command to run (ask, explain, review, test, index, web)")
    parser.add_argument("argument", nargs="?", default="", help="Argument for the command")
    parser.add_argument("--project", default="project", help="Target project directory")
    parser.add_argument("--force", action="store_true", help="Force re-indexing")

    args = parser.parse_args()

    config = load_config()
    vector_store = VectorStoreManager(config=config)
    orchestrator = WorkflowOrchestrator(config=config, vector_store=vector_store)
    patch_manager = PatchManager(config=config)

    if args.command == "interactive":
        run_interactive(orchestrator, patch_manager)
    elif args.command == "web":
        import subprocess
        console.print("[bold green]Launching Streamlit Web Interface...[/bold green]")
        subprocess.run(["uv", "run", "streamlit", "run", "app.py"])
    elif args.command == "index":
        target = args.argument or args.project
        res = orchestrator.indexer.index_project(target, force_reindex=args.force)
        console.print(res)
    elif args.command == "ask":
        if not args.argument:
            console.print("[red]Error: Missing question argument.[/red]")
            return
        res = orchestrator.ask(args.argument)
        console.print(Panel(Markdown(res.answer), title="DevMind Answer"))
    elif args.command == "explain":
        res = orchestrator.explain(args.argument)
        console.print(Panel(Markdown(res.answer), title="Explain"))
    elif args.command == "review":
        res = orchestrator.review(args.argument)
        console.print(Panel(Markdown(res.summary), title="Review"))
    elif args.command == "test":
        res = orchestrator.plan_tests(args.argument)
        console.print(Panel(Markdown(res.answer), title="Test Plan"))
    else:
        # Default: treat whole command as a question
        query = f"{args.command} {args.argument}".strip()
        res = orchestrator.ask(query)
        console.print(Panel(Markdown(res.answer), title="DevMind Answer"))


if __name__ == "__main__":
    main()
