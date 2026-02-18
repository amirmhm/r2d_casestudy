"""
Model evaluation and comparison for fine-tuned vs. base models.

Produces per-task breakdowns and an overall comparison report.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class TaskResult:
    """Per-task evaluation result for a single model."""

    task_id: str
    success: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0
    patch_applied: bool = False
    build_ok: bool = False
    checks_passed: bool = False


@dataclass
class ModelResults:
    """Aggregated results for a model."""

    model_name: str = ""
    task_results: list[TaskResult] = field(default_factory=list)
    pass_rate: float = 0.0
    total_tasks: int = 0
    tasks_passed: int = 0
    avg_duration: float = 0.0


@dataclass
class ComparisonReport:
    """Side-by-side comparison of two models."""

    base_model: ModelResults = field(default_factory=ModelResults)
    fine_tuned_model: ModelResults = field(default_factory=ModelResults)
    improvement_rate: float = 0.0
    per_task_comparison: list[dict] = field(default_factory=list)
    summary: str = ""


class ModelEvaluator:
    """Evaluates and compares model performance on evaluation tasks."""

    def evaluate_results(self, results: list[dict], model_name: str = "") -> ModelResults:
        """Parse raw evaluation results into a ModelResults summary.

        Each entry in *results* is expected to have keys like:
        ``task_id``, ``success``, ``error``, ``duration_seconds``,
        ``patch_applied``, ``build_ok``, ``checks_passed``.
        """
        model = ModelResults(model_name=model_name)
        model.total_tasks = len(results)

        total_duration = 0.0
        for r in results:
            tr = TaskResult(
                task_id=r.get("task_id", ""),
                success=r.get("success", False),
                error=r.get("error"),
                duration_seconds=r.get("duration_seconds", 0.0),
                patch_applied=r.get("patch_applied", False),
                build_ok=r.get("build_ok", False),
                checks_passed=r.get("checks_passed", False),
            )
            model.task_results.append(tr)
            if tr.success:
                model.tasks_passed += 1
            total_duration += tr.duration_seconds

        model.pass_rate = (
            (model.tasks_passed / model.total_tasks * 100.0) if model.total_tasks else 0.0
        )
        model.avg_duration = total_duration / max(model.total_tasks, 1)
        return model

    def generate_comparison(
        self,
        base_results: list[dict],
        finetuned_results: list[dict],
        base_name: str = "base",
        finetuned_name: str = "fine-tuned",
    ) -> ComparisonReport:
        """Generate a comparison report between base and fine-tuned models.

        Both result lists should be aligned (same task order).
        """
        base = self.evaluate_results(base_results, model_name=base_name)
        finetuned = self.evaluate_results(finetuned_results, model_name=finetuned_name)

        # Per-task comparison
        per_task: list[dict] = []
        base_by_task = {tr.task_id: tr for tr in base.task_results}
        ft_by_task = {tr.task_id: tr for tr in finetuned.task_results}

        all_task_ids = sorted(set(base_by_task.keys()) | set(ft_by_task.keys()))
        for task_id in all_task_ids:
            b = base_by_task.get(task_id)
            f = ft_by_task.get(task_id)
            per_task.append(
                {
                    "task_id": task_id,
                    "base_success": b.success if b else None,
                    "finetuned_success": f.success if f else None,
                    "base_duration": b.duration_seconds if b else None,
                    "finetuned_duration": f.duration_seconds if f else None,
                    "improved": (f.success if f else False) and not (b.success if b else False),
                    "regressed": (b.success if b else False) and not (f.success if f else False),
                }
            )

        improvement = finetuned.pass_rate - base.pass_rate
        improved_count = sum(1 for t in per_task if t.get("improved"))
        regressed_count = sum(1 for t in per_task if t.get("regressed"))

        summary = (
            f"Base pass rate: {base.pass_rate:.1f}%, "
            f"Fine-tuned pass rate: {finetuned.pass_rate:.1f}% "
            f"(delta: {improvement:+.1f}%). "
            f"{improved_count} task(s) improved, {regressed_count} regressed."
        )

        return ComparisonReport(
            base_model=base,
            fine_tuned_model=finetuned,
            improvement_rate=improvement,
            per_task_comparison=per_task,
            summary=summary,
        )

    def print_comparison(self, report: ComparisonReport) -> None:
        """Pretty-print a comparison report."""
        console.print(f"\n[bold]Model Comparison[/bold]")
        console.print(f"  Base ({report.base_model.model_name}): {report.base_model.pass_rate:.1f}% pass rate")
        console.print(
            f"  Fine-tuned ({report.fine_tuned_model.model_name}): "
            f"{report.fine_tuned_model.pass_rate:.1f}% pass rate"
        )
        console.print(f"  Improvement: {report.improvement_rate:+.1f}%")
        console.print(f"\n  Per-task breakdown:")
        for t in report.per_task_comparison:
            base_s = "PASS" if t["base_success"] else "FAIL"
            ft_s = "PASS" if t["finetuned_success"] else "FAIL"
            marker = ""
            if t.get("improved"):
                marker = " [green]+[/green]"
            elif t.get("regressed"):
                marker = " [red]-[/red]"
            console.print(f"    {t['task_id']}: {base_s} → {ft_s}{marker}")
        console.print(f"\n  {report.summary}")
