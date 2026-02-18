"""
Failure analysis: classifies evaluation failures, identifies patterns,
generates recommendations, and exports reports.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    GENERATION_ERROR = "GENERATION_ERROR"
    APPLY_FAILURE = "APPLY_FAILURE"
    BUILD_FAILURE = "BUILD_FAILURE"
    CHECK_FAILURE = "CHECK_FAILURE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassifiedFailure:
    """A single classified failure."""

    task_id: str
    category: FailureCategory
    error_message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class FailurePattern:
    """A recurring pattern across failures."""

    pattern_name: str
    count: int = 0
    affected_tasks: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FailureReport:
    """Full failure analysis report."""

    total_tasks: int = 0
    total_failures: int = 0
    total_successes: int = 0
    failure_rate: float = 0.0
    classified_failures: list[ClassifiedFailure] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    patterns: list[FailurePattern] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    correlation_analysis: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Classifies failures and generates actionable analysis reports."""

    def classify_failure(self, result: dict) -> Optional[ClassifiedFailure]:
        """Classify a single task result into a failure category.

        Returns ``None`` if the task succeeded.
        """
        if result.get("success", False):
            return None

        task_id = result.get("task_id", "unknown")
        error = result.get("error", "") or ""

        # Determine category based on available signals
        patch_applied = result.get("patch_applied", False)
        build_ok = result.get("build_ok", False)
        checks_passed = result.get("checks_passed", False)

        error_lower = error.lower()

        # Prioritise structural signals over keyword matching
        if not patch_applied:
            if "generation" in error_lower or "llm" in error_lower or "api" in error_lower:
                category = FailureCategory.GENERATION_ERROR
            elif "apply" in error_lower or "patch" in error_lower or "hunk" in error_lower:
                category = FailureCategory.APPLY_FAILURE
            elif "retrieval" in error_lower or "no relevant" in error_lower:
                category = FailureCategory.RETRIEVAL_MISS
            else:
                category = FailureCategory.APPLY_FAILURE
        elif not build_ok:
            category = FailureCategory.BUILD_FAILURE
        elif not checks_passed:
            category = FailureCategory.CHECK_FAILURE
        elif "retrieval" in error_lower or "no relevant" in error_lower:
            category = FailureCategory.RETRIEVAL_MISS
        else:
            category = FailureCategory.UNKNOWN

        return ClassifiedFailure(
            task_id=task_id,
            category=category,
            error_message=error,
            details={
                "patch_applied": patch_applied,
                "build_ok": build_ok,
                "checks_passed": checks_passed,
            },
        )

    def analyze_results(self, results: list[dict]) -> FailureReport:
        """Analyse a list of evaluation results and produce a report."""
        report = FailureReport()
        report.total_tasks = len(results)

        for r in results:
            if r.get("success", False):
                report.total_successes += 1
            else:
                report.total_failures += 1
                cf = self.classify_failure(r)
                if cf:
                    report.classified_failures.append(cf)

        report.failure_rate = (
            (report.total_failures / report.total_tasks * 100.0) if report.total_tasks else 0.0
        )

        # Category counts
        cat_counter: Counter[str] = Counter()
        for cf in report.classified_failures:
            cat_counter[cf.category.value] += 1
        report.category_counts = dict(cat_counter)

        # Patterns and recommendations
        report.patterns = self._identify_patterns(report.classified_failures)
        report.recommendations = self._generate_recommendations(report)
        report.correlation_analysis = self._correlation_analysis(results)

        return report

    def _identify_patterns(self, failures: list[ClassifiedFailure]) -> list[FailurePattern]:
        """Identify recurring patterns across classified failures."""
        patterns: list[FailurePattern] = []

        # Group by category
        by_category: dict[str, list[ClassifiedFailure]] = defaultdict(list)
        for cf in failures:
            by_category[cf.category.value].append(cf)

        for cat, items in by_category.items():
            patterns.append(
                FailurePattern(
                    pattern_name=f"Cluster: {cat}",
                    count=len(items),
                    affected_tasks=[i.task_id for i in items],
                    description=f"{len(items)} task(s) failed due to {cat.lower().replace('_', ' ')}.",
                )
            )

        # Check for common error substrings
        error_fragments: Counter[str] = Counter()
        for cf in failures:
            if cf.error_message:
                for keyword in ["timeout", "syntax", "type", "import", "export", "hunk", "patch"]:
                    if keyword in cf.error_message.lower():
                        error_fragments[keyword] += 1

        for fragment, count in error_fragments.most_common(5):
            if count >= 2:
                affected = [
                    cf.task_id for cf in failures if fragment in cf.error_message.lower()
                ]
                patterns.append(
                    FailurePattern(
                        pattern_name=f"Error keyword: '{fragment}'",
                        count=count,
                        affected_tasks=affected,
                        description=f"The keyword '{fragment}' appeared in {count} failure error messages.",
                    )
                )

        return patterns

    def _generate_recommendations(self, report: FailureReport) -> list[str]:
        """Generate actionable recommendations based on failure analysis."""
        recs: list[str] = []

        cats = report.category_counts

        if cats.get(FailureCategory.RETRIEVAL_MISS.value, 0) > 0:
            recs.append(
                "Improve retrieval quality: consider switching to hybrid strategy, "
                "adding AST-aware chunking, or tuning BM25 parameters."
            )

        if cats.get(FailureCategory.GENERATION_ERROR.value, 0) > 0:
            recs.append(
                "Address generation errors: check LLM endpoint health, "
                "review prompt templates, or try chain-of-thought prompting."
            )

        if cats.get(FailureCategory.APPLY_FAILURE.value, 0) > 0:
            recs.append(
                "Fix patch application failures: ensure diffs use correct file paths "
                "and line numbers. Consider adding diff validation before application."
            )

        if cats.get(FailureCategory.BUILD_FAILURE.value, 0) > 0:
            recs.append(
                "Resolve build failures: generated patches likely introduce TypeScript "
                "errors. Add type-checking hints to the prompt or use CoT reasoning."
            )

        if cats.get(FailureCategory.CHECK_FAILURE.value, 0) > 0:
            recs.append(
                "Address check failures: patches compile but don't pass acceptance "
                "criteria. Review task requirements and improve prompt specificity."
            )

        if report.failure_rate > 50:
            recs.append(
                "High failure rate (>50%): consider fine-tuning the model on the "
                "provided training examples to improve task-specific performance."
            )

        if not recs:
            recs.append("All tasks passed — no recommendations needed.")

        return recs

    def _correlation_analysis(self, results: list[dict]) -> dict:
        """Analyse correlation between retrieval quality and task success.

        Returns a dict with correlation insights.
        """
        analysis: dict = {}

        success_by_difficulty: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            difficulty = r.get("difficulty", r.get("metadata", {}).get("difficulty", "unknown"))
            success_by_difficulty[difficulty].append(r.get("success", False))

        difficulty_rates: dict[str, float] = {}
        for diff, successes in success_by_difficulty.items():
            if successes:
                difficulty_rates[diff] = sum(successes) / len(successes) * 100.0
        analysis["success_rate_by_difficulty"] = difficulty_rates

        # Retrieval quality vs success correlation
        retrieval_quality_success: list[float] = []
        retrieval_quality_failure: list[float] = []
        for r in results:
            rq = r.get("retrieval_score", r.get("retrieval_quality"))
            if rq is not None:
                if r.get("success", False):
                    retrieval_quality_success.append(float(rq))
                else:
                    retrieval_quality_failure.append(float(rq))

        if retrieval_quality_success:
            analysis["avg_retrieval_quality_success"] = (
                sum(retrieval_quality_success) / len(retrieval_quality_success)
            )
        if retrieval_quality_failure:
            analysis["avg_retrieval_quality_failure"] = (
                sum(retrieval_quality_failure) / len(retrieval_quality_failure)
            )

        return analysis

    def print_report(self, report: FailureReport) -> None:
        """Pretty-print a failure analysis report."""
        console.print(f"\n[bold]Failure Analysis Report[/bold]")
        console.print(f"  Total tasks: {report.total_tasks}")
        console.print(f"  Successes: {report.total_successes}")
        console.print(f"  Failures: {report.total_failures}")
        console.print(f"  Failure rate: {report.failure_rate:.1f}%")

        if report.category_counts:
            console.print(f"\n  [bold]Failure Categories:[/bold]")
            for cat, count in sorted(report.category_counts.items(), key=lambda x: -x[1]):
                console.print(f"    {cat}: {count}")

        if report.patterns:
            console.print(f"\n  [bold]Patterns:[/bold]")
            for p in report.patterns:
                console.print(f"    {p.pattern_name} ({p.count}x): {p.description}")

        if report.recommendations:
            console.print(f"\n  [bold]Recommendations:[/bold]")
            for i, rec in enumerate(report.recommendations, 1):
                console.print(f"    {i}. {rec}")

    def export_report(self, report: FailureReport, output_path: Path) -> None:
        """Export the failure report to a JSON file."""
        data = {
            "total_tasks": report.total_tasks,
            "total_failures": report.total_failures,
            "total_successes": report.total_successes,
            "failure_rate": report.failure_rate,
            "category_counts": report.category_counts,
            "classified_failures": [
                {
                    "task_id": cf.task_id,
                    "category": cf.category.value,
                    "error_message": cf.error_message,
                    "details": cf.details,
                }
                for cf in report.classified_failures
            ],
            "patterns": [
                {
                    "pattern_name": p.pattern_name,
                    "count": p.count,
                    "affected_tasks": p.affected_tasks,
                    "description": p.description,
                }
                for p in report.patterns
            ],
            "recommendations": report.recommendations,
            "correlation_analysis": report.correlation_analysis,
        }

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]Failure report exported to {output_path}[/green]")
