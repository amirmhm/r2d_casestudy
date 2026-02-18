"""
Training data curation for fine-tuning.

Loads examples from JSONL, validates quality, splits train/val
with optional stratification by task, and exports OpenAI chat format.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class TrainingExample:
    """A single training example."""

    task_id: str
    variant: str
    input_prompt: str
    expected_output: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """Summary of training data quality."""

    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    task_distribution: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass
class TrainValSplit:
    """Result of splitting training data."""

    train: list[TrainingExample] = field(default_factory=list)
    val: list[TrainingExample] = field(default_factory=list)
    train_size: int = 0
    val_size: int = 0


class DataCurator:
    """Curates and prepares training data for fine-tuning."""

    def load_examples(self, path: Path) -> list[TrainingExample]:
        """Load training examples from a JSONL file.

        Each line should be a JSON object with keys:
        ``task_id``, ``input_prompt``, ``expected_output``, ``metadata``.
        """
        examples: list[TrainingExample] = []
        content = path.read_text(encoding="utf-8").strip()
        for line_no, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(
                    TrainingExample(
                        task_id=obj.get("task_id", ""),
                        variant=obj.get("variant", ""),
                        input_prompt=obj.get("input_prompt", ""),
                        expected_output=obj.get("expected_output", ""),
                        metadata=obj.get("metadata", {}),
                    )
                )
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Skipping invalid JSON on line {line_no}: {e}[/yellow]")

        console.print(f"[blue]Loaded {len(examples)} examples from {path}[/blue]")
        return examples

    def validate_examples(self, examples: list[TrainingExample]) -> QualityReport:
        """Validate examples and produce a quality report."""
        report = QualityReport()
        report.total_examples = len(examples)

        total_input_tokens = 0
        total_output_tokens = 0

        for ex in examples:
            issues: list[str] = []
            if not ex.task_id:
                issues.append("missing task_id")
            if not ex.input_prompt:
                issues.append("missing input_prompt")
            if not ex.expected_output:
                issues.append("missing expected_output")

            if issues:
                report.invalid_examples += 1
                report.issues.extend(
                    f"{ex.task_id or 'unknown'}/{ex.variant}: {i}" for i in issues
                )
            else:
                report.valid_examples += 1

            # Rough token count (words ÷ 0.75)
            input_words = len(ex.input_prompt.split())
            output_words = len(ex.expected_output.split())
            total_input_tokens += input_words
            total_output_tokens += output_words

            # Distributions
            report.task_distribution[ex.task_id] = (
                report.task_distribution.get(ex.task_id, 0) + 1
            )
            quality = ex.metadata.get("quality", "unknown")
            report.quality_distribution[quality] = (
                report.quality_distribution.get(quality, 0) + 1
            )

        n = max(report.total_examples, 1)
        report.avg_input_tokens = total_input_tokens / n
        report.avg_output_tokens = total_output_tokens / n

        return report

    def split_train_val(
        self,
        examples: list[TrainingExample],
        val_ratio: float = 0.2,
        stratify_by_task: bool = True,
        seed: int = 42,
    ) -> TrainValSplit:
        """Split examples into train and validation sets.

        When ``stratify_by_task`` is True, the split is performed per-task
        to ensure each task is represented in both train and val sets.
        """
        rng = random.Random(seed)

        if not stratify_by_task:
            shuffled = list(examples)
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
            train = shuffled[:split_idx]
            val = shuffled[split_idx:]
        else:
            by_task: dict[str, list[TrainingExample]] = defaultdict(list)
            for ex in examples:
                by_task[ex.task_id].append(ex)

            train: list[TrainingExample] = []
            val: list[TrainingExample] = []

            for task_id in sorted(by_task.keys()):
                task_examples = by_task[task_id]
                rng.shuffle(task_examples)
                n_val = max(1, int(len(task_examples) * val_ratio))
                val.extend(task_examples[:n_val])
                train.extend(task_examples[n_val:])

        return TrainValSplit(
            train=train,
            val=val,
            train_size=len(train),
            val_size=len(val),
        )

    def format_for_openai(
        self,
        examples: list[TrainingExample],
        include_quality: Optional[str] = None,
    ) -> list[dict]:
        """Format examples into OpenAI chat fine-tuning format.

        Each entry becomes:
        ``{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}``

        If ``include_quality`` is set (e.g. ``"gold"``), only examples with
        that quality label are included.
        """
        formatted: list[dict] = []

        for ex in examples:
            if include_quality:
                quality = ex.metadata.get("quality", "")
                if quality != include_quality:
                    continue

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert TypeScript developer. Generate minimal "
                        "unified diff patches for a TypeScript game codebase. "
                        "Output ONLY the raw unified diff."
                    ),
                },
                {"role": "user", "content": ex.input_prompt},
                {"role": "assistant", "content": ex.expected_output},
            ]
            formatted.append({"messages": messages})

        return formatted

    def export_jsonl(self, data: list[dict], output_path: Path) -> None:
        """Export formatted data to a JSONL file."""
        with output_path.open("w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        console.print(f"[green]Exported {len(data)} examples to {output_path}[/green]")
