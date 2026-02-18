"""
CLI interface using Typer.
Provides commands: index, run-eval, run-task.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import get_settings

app = typer.Typer(
    name="ggf-case",
    help="GGF LLM Systems Case - Evaluation CLI",
    add_completion=False,
)
console = Console()


def resolve_repo_root() -> Path:
    """Find the repository root (parent of 'solution' directory)."""
    current = Path.cwd()

    # Check if we're in the repo root
    if (current / "eval" / "tasks.json").exists():
        return current

    # Check if we're in solution/
    parent = current.parent
    if (parent / "eval" / "tasks.json").exists():
        return parent

    # Fallback
    console.print("[yellow]Warning: Could not auto-detect repo root. Using current directory.[/yellow]")
    return current


@app.command()
def index(
    source_dir: Optional[str] = typer.Option(None, help="Source directory to index"),
    output: str = typer.Option("index.json", help="Output index file path"),
) -> None:
    """Index the mini-game codebase for RAG retrieval."""
    from .rag.indexer import index_codebase, save_index

    repo_root = resolve_repo_root()
    src = Path(source_dir) if source_dir else repo_root / "ggf-mini-game" / "src"
    out = Path(output)

    console.print(f"[bold]Indexing codebase from {src}[/bold]")
    idx = index_codebase(src)
    save_index(idx, out)
    console.print(f"[green]Index saved with {len(idx.chunks)} chunks[/green]")


@app.command(name="run-eval")
def run_eval(
    output_dir: str = typer.Option("eval/outputs", help="Output directory for results"),
    tasks: Optional[str] = typer.Option(None, help="Comma-separated task IDs to run (default: all)"),
) -> None:
    """Run the full evaluation suite over all tasks."""
    from .eval.runner import run_evaluation

    repo_root = resolve_repo_root()
    settings = get_settings(repo_root=repo_root)

    if not settings.openai_api_key:
        console.print("[red]ERROR: OPENAI_API_KEY is not set.[/red]")
        console.print("Set it in .env file or as environment variable.")
        raise typer.Exit(code=1)

    task_filter = [t.strip() for t in tasks.split(",")] if tasks else None
    out = repo_root / output_dir

    summary = run_evaluation(settings, repo_root, out, task_filter)

    if summary.pass_rate < 100:
        raise typer.Exit(code=1)


@app.command(name="run-task")
def run_task(
    task_id: str = typer.Argument(help="Task ID to run (e.g., task_01)"),
    output_dir: str = typer.Option("eval/outputs", help="Output directory"),
) -> None:
    """Run a single evaluation task."""
    from .eval.runner import run_evaluation

    repo_root = resolve_repo_root()
    settings = get_settings(repo_root=repo_root)

    if not settings.openai_api_key:
        console.print("[red]ERROR: OPENAI_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)

    out = repo_root / output_dir
    summary = run_evaluation(settings, repo_root, out, [task_id])

    if not summary.results or not summary.results[0].success:
        raise typer.Exit(code=1)


@app.command()
def check_health() -> None:
    """Check if the LLM endpoint is reachable."""
    from .llm.openai_compat import LLMClient

    settings = get_settings()
    client = LLMClient(settings)

    if client.health_check():
        console.print(f"[green]LLM endpoint is healthy: {settings.openai_base_url}[/green]")
    else:
        console.print(f"[red]LLM endpoint is not reachable: {settings.openai_base_url}[/red]")
        raise typer.Exit(code=1)


# --- Phase 2: New Commands ---

@app.command()
def metrics(
    index_path: str = typer.Option("index.json", help="Path to codebase index"),
    gold_labels_path: Optional[str] = typer.Option(None, help="Path to gold labels JSON"),
) -> None:
    """Compute and display retrieval metrics against gold labels."""
    import json
    from .rag.indexer import load_index
    from .rag.retriever import retrieve
    from .metrics.retrieval_metrics import compute_retrieval_scores

    repo_root = resolve_repo_root()
    settings = get_settings(repo_root=repo_root)

    idx_path = Path(index_path)
    if not idx_path.exists():
        console.print("[yellow]Index not found, building...[/yellow]")
        from .rag.indexer import index_codebase, save_index
        src = repo_root / "ggf-mini-game" / "src"
        index = index_codebase(src, strategy=settings.chunk_strategy)
        save_index(index, idx_path)
    else:
        index = load_index(idx_path)

    gl_path = Path(gold_labels_path) if gold_labels_path else repo_root / "eval" / "gold_labels.json"
    if not gl_path.exists():
        console.print("[red]Gold labels not found. Cannot compute metrics.[/red]")
        raise typer.Exit(code=1)

    gold = json.loads(gl_path.read_text(encoding="utf-8"))
    queries = []
    for task_id, task_gold in gold.get("tasks", {}).items():
        relevant_files = task_gold.get("relevant_files_ranked", [])
        results = retrieve(index, " ".join(relevant_files), top_k=settings.top_k,
                          strategy=settings.retrieval_strategy)
        retrieved_files = [r.chunk.file_path for r in results]
        queries.append({"retrieved": retrieved_files, "relevant": relevant_files})

    scores = compute_retrieval_scores(queries, k=5)
    console.print(f"\n[bold]Retrieval Metrics (k=5, strategy={settings.retrieval_strategy}):[/bold]")
    console.print(f"  Precision@5: {scores.precision_at_k:.3f}")
    console.print(f"  Recall@5:    {scores.recall_at_k:.3f}")
    console.print(f"  MRR:         {scores.mrr:.3f}")
    console.print(f"  NDCG@5:      {scores.ndcg_at_k:.3f}")
    console.print(f"  Hit Rate:    {scores.hit_rate:.3f}")
    console.print(f"  Queries:     {scores.num_queries}")


# --- Fine-tuning commands ---

finetune_app = typer.Typer(help="Fine-tuning commands")
app.add_typer(finetune_app, name="finetune")


@finetune_app.command()
def prepare(
    output: str = typer.Option("finetune_train.jsonl", help="Output JSONL file"),
    quality: str = typer.Option("gold", help="Quality filter (gold, all)"),
) -> None:
    """Prepare training data for fine-tuning."""
    from .finetune.data_curator import DataCurator

    repo_root = resolve_repo_root()
    examples_path = repo_root / "eval" / "training_data" / "examples.jsonl"

    curator = DataCurator()
    examples = curator.load_examples(examples_path)
    report = curator.validate_examples(examples)

    console.print(f"[bold]Training Data Quality Report:[/bold]")
    console.print(f"  Total examples: {report.total_examples}")
    console.print(f"  Valid: {report.valid_examples}")
    console.print(f"  Invalid: {report.invalid_examples}")
    console.print(f"  Avg input tokens: {report.avg_input_tokens:.0f}")
    console.print(f"  Avg output tokens: {report.avg_output_tokens:.0f}")
    console.print(f"  Task distribution: {report.task_distribution}")
    console.print(f"  Quality distribution: {report.quality_distribution}")

    split = curator.split_train_val(examples, stratify_by_task=True)
    console.print(f"\n[bold]Split:[/bold] {split.train_size} train, {split.val_size} val")

    quality_filter = quality if quality != "all" else None
    formatted = curator.format_for_openai(split.train, include_quality=quality_filter)
    curator.export_jsonl(formatted, Path(output))


@finetune_app.command(name="run")
def finetune_run(
    training_file: str = typer.Argument(help="Path to training JSONL"),
    model: str = typer.Option("gpt-4o-mini-2024-07-18", help="Base model to fine-tune"),
    suffix: str = typer.Option("ggf-case", help="Model suffix"),
) -> None:
    """Start a fine-tuning job via OpenAI API."""
    from .finetune.trainer import FineTuneTrainer, FineTuneConfig

    settings = get_settings()
    if not settings.openai_api_key:
        console.print("[red]ERROR: OPENAI_API_KEY required for fine-tuning.[/red]")
        raise typer.Exit(code=1)

    trainer = FineTuneTrainer(settings)
    config = FineTuneConfig(model=model, suffix=suffix)

    file_id = trainer.upload_training_file(Path(training_file))
    job = trainer.create_job(file_id, config)

    console.print(f"\n[bold]Job started: {job.job_id}[/bold]")
    console.print(f"Status: {job.status}")
    console.print(f"Monitor with: ggf-case finetune eval --job-id {job.job_id}")


@finetune_app.command(name="eval")
def finetune_eval(
    job_id: Optional[str] = typer.Option(None, help="Fine-tuning job ID to check"),
) -> None:
    """Check fine-tuning job status or evaluate fine-tuned model."""
    from .finetune.trainer import FineTuneTrainer

    settings = get_settings()
    trainer = FineTuneTrainer(settings)

    if job_id:
        job = trainer.get_job_status(job_id)
        console.print(f"[bold]Job {job.job_id}:[/bold]")
        console.print(f"  Status: {job.status}")
        console.print(f"  Model: {job.model}")
        if job.fine_tuned_model:
            console.print(f"  Fine-tuned model: {job.fine_tuned_model}")
        if job.error:
            console.print(f"  Error: {job.error}")
    else:
        jobs = trainer.list_jobs(limit=5)
        for j in jobs:
            status_color = "green" if j.status == "succeeded" else "yellow" if j.status == "running" else "red"
            console.print(f"  [{status_color}]{j.job_id}[/{status_color}] - {j.status} ({j.model})")


# --- Analytics commands ---

@app.command(name="analyze")
def analyze_failures(
    results_path: str = typer.Argument(help="Path to evaluation summary.json"),
    output: str = typer.Option("failure_analysis.json", help="Output file"),
) -> None:
    """Run failure analysis on evaluation results."""
    import json
    from .analytics.failure_analyzer import FailureAnalyzer

    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    results = data.get("results", [])

    analyzer = FailureAnalyzer()
    report = analyzer.analyze_results(results)
    analyzer.print_report(report)
    analyzer.export_report(report, Path(output))


@app.command(name="report")
def generate_report(
    results_dir: str = typer.Argument(help="Path to evaluation run directory"),
) -> None:
    """Generate a comprehensive evaluation report."""
    import json

    run_dir = Path(results_dir)
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        console.print(f"[red]No summary.json found in {run_dir}[/red]")
        raise typer.Exit(code=1)

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    console.print(f"\n[bold]== Evaluation Report ==[/bold]")
    console.print(f"Timestamp: {data.get('timestamp')}")
    console.print(f"Total Tasks: {data.get('total_tasks')}")
    console.print(f"Passed: {data.get('tasks_passed')}")
    console.print(f"Failed: {data.get('tasks_failed')}")
    console.print(f"Pass Rate: {data.get('pass_rate')}%")
    console.print(f"Duration: {data.get('total_duration_seconds')}s")

    # Run failure analysis
    from .analytics.failure_analyzer import FailureAnalyzer
    analyzer = FailureAnalyzer()
    report = analyzer.analyze_results(data.get("results", []))
    analyzer.print_report(report)

    # Save
    output = run_dir / "full_report.json"
    analyzer.export_report(report, output)
    console.print(f"\n[green]Full report saved to {output}[/green]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
