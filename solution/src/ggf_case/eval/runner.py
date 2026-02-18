"""
Evaluation runner.
Loops over tasks, calls RAG retrieval, generates patches via LLM,
applies patches, runs checks, and collects metrics.
"""

import json
import subprocess
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from ..config import Settings
from ..rag.indexer import CodebaseIndex, index_codebase, save_index, load_index
from ..rag.retriever import retrieve, format_context
from ..llm.openai_compat import LLMClient
from ..llm.prompts import build_patch_prompt
from ..patch.diff_guard import check_diff, extract_diff_from_response
from ..patch.apply_patch import apply_patch, create_working_copy

console = Console()


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    title: str
    success: bool
    check_passed: bool
    patch_generated: bool
    patch_applied: bool
    guard_passed: bool
    error: str = ""
    duration_seconds: float = 0.0
    diff_stats: dict = field(default_factory=dict)
    retrieval_count: int = 0


@dataclass
class EvalSummary:
    """Summary of the full evaluation run."""
    timestamp: str
    total_tasks: int
    tasks_passed: int
    tasks_failed: int
    pass_rate: float
    total_duration_seconds: float
    results: list[TaskResult] = field(default_factory=list)


def load_tasks(tasks_path: Path) -> list[dict]:
    """Load tasks from tasks.json."""
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    return data.get("tasks", [])


def run_build(working_dir: Path) -> tuple[bool, str]:
    """Run npm build in the working directory."""
    try:
        result = subprocess.run(
            "npm run build",
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,  # Required on Windows for npm (.cmd)
        )
        if result.returncode == 0:
            return True, "Build succeeded"
        return False, f"Build failed: {result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, "Build timed out"
    except Exception as e:
        return False, f"Build error: {e}"


def run_check(
    task_id: str,
    working_dir: Path,
    repo_root: Path,
    check_script: Path,
) -> tuple[bool, str]:
    """Run the check script for a task."""
    try:
        result = subprocess.run(
            f"node {check_script} --task {task_id} --workdir {working_dir}",
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,  # Required on Windows for npm (.cmd)
        )
        if result.returncode == 0:
            return True, "Check passed"
        output = (result.stdout + result.stderr)[:1000]
        return False, f"Check failed: {output}"
    except subprocess.TimeoutExpired:
        return False, "Check timed out"
    except Exception as e:
        return False, f"Check error: {e}"


def run_single_task(
    task: dict,
    index: CodebaseIndex,
    llm_client: LLMClient,
    working_dir: Path,
    repo_root: Path,
    settings: Settings,
) -> TaskResult:
    """
    Run a single evaluation task.

    Steps:
    1. Retrieve relevant code context
    2. Generate patch via LLM
    3. Validate patch with diff guard
    4. Apply patch to working copy
    5. Build the project
    6. Run acceptance check
    """
    task_id = task["id"]
    title = task["title"]
    start_time = time.time()

    result = TaskResult(task_id=task_id, title=title, success=False,
                        check_passed=False, patch_generated=False,
                        patch_applied=False, guard_passed=False)

    try:
        # Step 1: Retrieve context
        console.print(f"  [blue]Retrieving context for {task_id}...[/blue]")
        query = f"{task['user_request']} {' '.join(task.get('suggested_files', []))}"
        retrieval_results = retrieve(
            index, query,
            top_k=settings.top_k,
            file_filter=task.get("suggested_files"),
        )
        result.retrieval_count = len(retrieval_results)
        context = format_context(retrieval_results)

        # Step 2: Generate patch
        console.print(f"  [blue]Generating patch via LLM...[/blue]")
        messages = build_patch_prompt(
            task_title=title,
            user_request=task["user_request"],
            acceptance_criteria=task["acceptance_criteria"],
            suggested_files=task.get("suggested_files", []),
            code_context=context,
        )
        raw_response = llm_client.chat_completion(messages)
        diff_text = extract_diff_from_response(raw_response)
        result.patch_generated = bool(diff_text.strip())

        if not result.patch_generated:
            result.error = "LLM returned empty patch"
            return result

        # Step 3: Diff guard
        console.print(f"  [blue]Checking diff guard...[/blue]")
        guard = check_diff(
            diff_text,
            max_lines=settings.diff_max_lines,
            max_files=settings.diff_max_files,
        )
        result.guard_passed = guard.passed
        result.diff_stats = {
            "files_changed": guard.stats.files_changed,
            "lines_added": guard.stats.lines_added,
            "lines_removed": guard.stats.lines_removed,
            "total_changed": guard.stats.total_changed,
        }

        if not guard.passed:
            result.error = f"Diff guard: {guard.reason}"
            return result

        # Step 4: Apply patch
        console.print(f"  [blue]Applying patch...[/blue]")
        patch_result = apply_patch(diff_text, working_dir)
        result.patch_applied = patch_result.success

        if not patch_result.success:
            result.error = f"Patch apply: {patch_result.message}"
            return result

        # Step 5: Build
        console.print(f"  [blue]Building project...[/blue]")
        build_ok, build_msg = run_build(working_dir)
        if not build_ok:
            result.error = f"Build: {build_msg}"
            return result

        # Step 6: Run check
        console.print(f"  [blue]Running acceptance check...[/blue]")
        check_script = repo_root / "eval" / "checks" / "run_check.mjs"
        check_ok, check_msg = run_check(task_id, working_dir, repo_root, check_script)
        result.check_passed = check_ok
        result.success = check_ok

        if not check_ok:
            result.error = f"Check: {check_msg}"

    except Exception as e:
        result.error = str(e)
    finally:
        result.duration_seconds = round(time.time() - start_time, 2)

    return result


def run_evaluation(
    settings: Settings,
    repo_root: Path,
    output_dir: Path,
    task_filter: list[str] | None = None,
) -> EvalSummary:
    """
    Run the full evaluation suite.

    Args:
        settings: Application settings.
        repo_root: Root of the case repository.
        output_dir: Directory to write outputs.
        task_filter: Optional list of task IDs to run (None = all).

    Returns:
        EvalSummary with all results.
    """
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Setup output directory
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks_path = repo_root / "eval" / "tasks.json"
    tasks = load_tasks(tasks_path)
    if task_filter:
        tasks = [t for t in tasks if t["id"] in task_filter]

    console.print(f"\n[bold]Running {len(tasks)} tasks[/bold]\n")

    # Index codebase
    console.print("[bold blue]Step 1: Indexing codebase...[/bold blue]")
    mini_game_dir = repo_root / "ggf-mini-game"
    index_path = run_dir / "index.json"

    index = index_codebase(mini_game_dir / "src")
    save_index(index, index_path)

    # Initialize LLM client
    console.print("[bold blue]Step 2: Initializing LLM client...[/bold blue]")
    llm_client = LLMClient(settings)

    # Run tasks
    results: list[TaskResult] = []
    for i, task in enumerate(tasks):
        console.print(f"\n[bold yellow]Task {i+1}/{len(tasks)}: {task['id']} - {task['title']}[/bold yellow]")

        # Create fresh working copy for each task
        work_dir = run_dir / f"{task['id']}_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        create_working_copy(mini_game_dir, work_dir)
        actual_work = work_dir / "work"

        # Install deps if needed
        npm_install(actual_work)

        task_result = run_single_task(
            task, index, llm_client, actual_work, repo_root, settings
        )
        results.append(task_result)

        status = "[green]PASS[/green]" if task_result.success else "[red]FAIL[/red]"
        console.print(f"  Result: {status} ({task_result.duration_seconds}s)")
        if task_result.error:
            console.print(f"  Error: {task_result.error}")

    # Summary
    passed = sum(1 for r in results if r.success)
    total = len(results)
    total_duration = round(time.time() - start_time, 2)

    summary = EvalSummary(
        timestamp=timestamp,
        total_tasks=total,
        tasks_passed=passed,
        tasks_failed=total - passed,
        pass_rate=round(passed / total * 100, 1) if total > 0 else 0,
        total_duration_seconds=total_duration,
        results=results,
    )

    # Write outputs
    write_outputs(summary, run_dir)
    print_summary_table(summary)

    return summary


def npm_install(working_dir: Path) -> None:
    """Run npm install in a directory."""
    try:
        subprocess.run(
            "npm install",
            cwd=str(working_dir),
            capture_output=True,
            timeout=60,
            shell=True,  # Required on Windows for npm (.cmd)
        )
    except Exception:
        pass


def write_outputs(summary: EvalSummary, run_dir: Path) -> None:
    """Write evaluation outputs to files."""
    # Summary JSON
    summary_path = run_dir / "summary.json"
    summary_data = {
        "timestamp": summary.timestamp,
        "total_tasks": summary.total_tasks,
        "tasks_passed": summary.tasks_passed,
        "tasks_failed": summary.tasks_failed,
        "pass_rate": summary.pass_rate,
        "total_duration_seconds": summary.total_duration_seconds,
        "results": [asdict(r) for r in summary.results],
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # JSONL log
    log_path = run_dir / "logs.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in summary.results:
            f.write(json.dumps(asdict(r)) + "\n")

    console.print(f"\n[green]Outputs written to {run_dir}[/green]")


def print_summary_table(summary: EvalSummary) -> None:
    """Print a rich summary table."""
    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Error", style="red", max_width=40)

    for r in summary.results:
        status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        table.add_row(r.task_id, r.title, status, f"{r.duration_seconds}s", r.error[:40] if r.error else "")

    console.print(table)
    console.print(f"\n[bold]Pass Rate: {summary.pass_rate}% ({summary.tasks_passed}/{summary.total_tasks})[/bold]")
    console.print(f"[bold]Total Time: {summary.total_duration_seconds}s[/bold]")
