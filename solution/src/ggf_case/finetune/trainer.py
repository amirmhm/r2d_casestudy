"""
Fine-tuning orchestration via the OpenAI /v1/fine_tuning/jobs API.

Handles file upload, job creation, status polling, and job listing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

from ..config import Settings

console = Console()


@dataclass
class FineTuneConfig:
    """Hyperparameters for a fine-tuning job.

    Attributes:
        model: Base model to fine-tune (e.g. ``gpt-4o-mini-2024-07-18``).
        suffix: Custom model name suffix.
        n_epochs: Number of training epochs (``"auto"`` lets OpenAI decide).
        batch_size: Training batch size (``"auto"`` lets OpenAI decide).
        learning_rate_multiplier: Scaling factor for the learning rate
            (``"auto"`` lets OpenAI decide).
    """

    model: str = "gpt-4o-mini-2024-07-18"
    suffix: str = "ggf-case"
    n_epochs: int | str = "auto"
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"


@dataclass
class FineTuneJob:
    """Represents a fine-tuning job."""

    job_id: str = ""
    status: str = ""
    model: str = ""
    fine_tuned_model: Optional[str] = None
    error: Optional[str] = None
    created_at: int = 0
    finished_at: Optional[int] = None
    trained_tokens: Optional[int] = None
    hyperparameters: dict = field(default_factory=dict)


class FineTuneTrainer:
    """Manages fine-tuning jobs via the OpenAI API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.openai_api_key

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def upload_training_file(self, file_path: Path) -> str:
        """Upload a JSONL training file to the OpenAI files endpoint.

        Returns the file ID (e.g. ``file-abc123``).
        """
        url = f"{self.base_url}/files"

        with file_path.open("rb") as f:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    url,
                    headers=self._headers(),
                    files={"file": (file_path.name, f, "application/jsonl")},
                    data={"purpose": "fine-tune"},
                )
                response.raise_for_status()
                data = response.json()

        file_id = data.get("id", "")
        console.print(f"[green]Uploaded {file_path.name} → {file_id}[/green]")
        return file_id

    def create_job(self, training_file_id: str, config: FineTuneConfig) -> FineTuneJob:
        """Create a fine-tuning job.

        Args:
            training_file_id: ID of the uploaded training file.
            config: Hyperparameter configuration.

        Returns:
            A :class:`FineTuneJob` with the initial status.
        """
        url = f"{self.base_url}/fine_tuning/jobs"

        hyperparameters: dict = {}
        if config.n_epochs != "auto":
            hyperparameters["n_epochs"] = config.n_epochs
        if config.batch_size != "auto":
            hyperparameters["batch_size"] = config.batch_size
        if config.learning_rate_multiplier != "auto":
            hyperparameters["learning_rate_multiplier"] = config.learning_rate_multiplier

        payload: dict = {
            "training_file": training_file_id,
            "model": config.model,
            "suffix": config.suffix,
        }
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        job = self._parse_job(data)
        console.print(f"[green]Created fine-tuning job: {job.job_id}[/green]")
        return job

    def get_job_status(self, job_id: str) -> FineTuneJob:
        """Retrieve the current status of a fine-tuning job."""
        url = f"{self.base_url}/fine_tuning/jobs/{job_id}"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()

        return self._parse_job(data)

    def list_jobs(self, limit: int = 10) -> list[FineTuneJob]:
        """List recent fine-tuning jobs."""
        url = f"{self.base_url}/fine_tuning/jobs"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                headers=self._headers(),
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()

        jobs: list[FineTuneJob] = []
        for item in data.get("data", []):
            jobs.append(self._parse_job(item))
        return jobs

    @staticmethod
    def _parse_job(data: dict) -> FineTuneJob:
        """Parse API response dict into a FineTuneJob."""
        error_obj = data.get("error")
        error_msg = None
        if error_obj and isinstance(error_obj, dict):
            error_msg = error_obj.get("message")
        elif isinstance(error_obj, str):
            error_msg = error_obj

        return FineTuneJob(
            job_id=data.get("id", ""),
            status=data.get("status", ""),
            model=data.get("model", ""),
            fine_tuned_model=data.get("fine_tuned_model"),
            error=error_msg,
            created_at=data.get("created_at", 0),
            finished_at=data.get("finished_at"),
            trained_tokens=data.get("trained_tokens"),
            hyperparameters=data.get("hyperparameters", {}),
        )
