"""
A/B experiment framework with paired t-test and Cohen's d effect size.

Compares two experiment variants (e.g. base vs. fine-tuned, keyword vs. hybrid)
and produces a statistical report.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str = ""
    variant_a_name: str = "control"
    variant_b_name: str = "treatment"
    metric_name: str = "pass_rate"
    n_runs: int = 5
    significance_level: float = 0.05


@dataclass
class StatisticalResult:
    """Result of a statistical significance test."""

    t_stat: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    is_significant: bool = False
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class ExperimentReport:
    """Full experiment comparison report."""

    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    variant_a_scores: list[float] = field(default_factory=list)
    variant_b_scores: list[float] = field(default_factory=list)
    variant_a_mean: float = 0.0
    variant_b_mean: float = 0.0
    variant_a_std: float = 0.0
    variant_b_std: float = 0.0
    statistical_result: StatisticalResult = field(default_factory=StatisticalResult)
    summary: str = ""


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], ddof: int = 1) -> float:
    if len(values) <= ddof:
        return 0.0
    m = _mean(values)
    ss = sum((x - m) ** 2 for x in values)
    return math.sqrt(ss / (len(values) - ddof))


def _paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Compute a paired t-test statistic and two-tailed p-value.

    Uses the difference of paired observations.
    Returns ``(t_stat, p_value)``.  The p-value is approximated
    using the normal distribution for simplicity (exact t-distribution
    requires scipy which may not be installed).
    """
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 1.0

    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = _mean(diffs)
    std_d = _std(diffs, ddof=1)

    if std_d == 0:
        return 0.0, 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))

    # Approximate p-value via normal CDF (good enough for n >= 5)
    p_value = 2.0 * _normal_cdf(-abs(t_stat))
    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = _mean(a)
    mean_b = _mean(b)
    std_a = _std(a, ddof=1)
    std_b = _std(b, ddof=1)

    # Pooled standard deviation
    pooled_std = math.sqrt(
        ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs A/B experiments and generates statistical reports."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        self.config = config or ExperimentConfig()

    def run_comparison(
        self,
        variant_a_scores: list[float],
        variant_b_scores: list[float],
    ) -> StatisticalResult:
        """Compute statistical comparison between two sets of scores.

        Uses paired t-test and Cohen's d effect size.
        """
        t_stat, p_value = _paired_t_test(variant_a_scores, variant_b_scores)
        d = _cohens_d(variant_a_scores, variant_b_scores)

        # Confidence interval for mean difference
        n = min(len(variant_a_scores), len(variant_b_scores))
        diffs = [
            variant_a_scores[i] - variant_b_scores[i] for i in range(n)
        ]
        mean_diff = _mean(diffs)
        std_diff = _std(diffs, ddof=1)
        margin = 1.96 * std_diff / math.sqrt(max(n, 1))  # ~95% CI

        return StatisticalResult(
            t_stat=t_stat,
            p_value=p_value,
            cohens_d=d,
            is_significant=p_value < self.config.significance_level,
            confidence_interval=(mean_diff - margin, mean_diff + margin),
        )

    def generate_report(
        self,
        variant_a_scores: list[float],
        variant_b_scores: list[float],
    ) -> ExperimentReport:
        """Generate a full experiment report."""
        stat = self.run_comparison(variant_a_scores, variant_b_scores)

        mean_a = _mean(variant_a_scores)
        mean_b = _mean(variant_b_scores)
        std_a = _std(variant_a_scores)
        std_b = _std(variant_b_scores)

        sig_str = "statistically significant" if stat.is_significant else "not statistically significant"
        effect_label = "negligible"
        abs_d = abs(stat.cohens_d)
        if abs_d >= 0.8:
            effect_label = "large"
        elif abs_d >= 0.5:
            effect_label = "medium"
        elif abs_d >= 0.2:
            effect_label = "small"

        summary = (
            f"{self.config.variant_a_name} mean={mean_a:.3f} (std={std_a:.3f}), "
            f"{self.config.variant_b_name} mean={mean_b:.3f} (std={std_b:.3f}). "
            f"Difference is {sig_str} (p={stat.p_value:.4f}). "
            f"Effect size (Cohen's d): {stat.cohens_d:.3f} ({effect_label})."
        )

        return ExperimentReport(
            config=self.config,
            variant_a_scores=variant_a_scores,
            variant_b_scores=variant_b_scores,
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            variant_a_std=std_a,
            variant_b_std=std_b,
            statistical_result=stat,
            summary=summary,
        )

    def export_report(self, report: ExperimentReport, output_path: Path) -> None:
        """Export experiment report to JSON."""
        data = {
            "config": {
                "name": report.config.name,
                "variant_a_name": report.config.variant_a_name,
                "variant_b_name": report.config.variant_b_name,
                "metric_name": report.config.metric_name,
                "n_runs": report.config.n_runs,
                "significance_level": report.config.significance_level,
            },
            "variant_a_scores": report.variant_a_scores,
            "variant_b_scores": report.variant_b_scores,
            "variant_a_mean": report.variant_a_mean,
            "variant_b_mean": report.variant_b_mean,
            "variant_a_std": report.variant_a_std,
            "variant_b_std": report.variant_b_std,
            "statistical_result": {
                "t_stat": report.statistical_result.t_stat,
                "p_value": report.statistical_result.p_value,
                "cohens_d": report.statistical_result.cohens_d,
                "is_significant": report.statistical_result.is_significant,
                "confidence_interval": list(report.statistical_result.confidence_interval),
            },
            "summary": report.summary,
        }

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]Experiment report exported to {output_path}[/green]")

    def print_report(self, report: ExperimentReport) -> None:
        """Pretty-print the experiment report."""
        console.print(f"\n[bold]Experiment: {report.config.name or 'A/B Comparison'}[/bold]")
        console.print(
            f"  {report.config.variant_a_name}: "
            f"mean={report.variant_a_mean:.3f}, std={report.variant_a_std:.3f}"
        )
        console.print(
            f"  {report.config.variant_b_name}: "
            f"mean={report.variant_b_mean:.3f}, std={report.variant_b_std:.3f}"
        )
        sr = report.statistical_result
        console.print(f"\n  t-statistic: {sr.t_stat:.4f}")
        console.print(f"  p-value:     {sr.p_value:.4f}")
        console.print(f"  Cohen's d:   {sr.cohens_d:.4f}")
        console.print(f"  Significant: {sr.is_significant}")
        console.print(f"  95% CI:      ({sr.confidence_interval[0]:.4f}, {sr.confidence_interval[1]:.4f})")
        console.print(f"\n  {report.summary}")
