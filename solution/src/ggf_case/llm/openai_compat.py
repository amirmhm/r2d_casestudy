"""
OpenAI-compatible LLM client.
Works with vLLM, Ollama, OpenAI, and any compatible endpoint.
"""

import httpx
from typing import Optional

from rich.console import Console

from ..config import Settings

console = Console()


def _use_completion_tokens(model: str) -> bool:
    """Check if the model requires max_completion_tokens instead of max_tokens."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in m


class LLMClient:
    """Client for OpenAI-compatible chat completion APIs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model

        if not self.api_key:
            console.print(
                "[red]WARNING: OPENAI_API_KEY is not set. "
                "LLM calls will fail. Set it in .env or environment.[/red]"
            )

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )

        url = f"{self.base_url}/chat/completions"
        # Newer models (gpt-5, o-series) require max_completion_tokens
        token_key = "max_completion_tokens" if _use_completion_tokens(self.model) else "max_tokens"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            token_key: max_tokens,
        }
        if stop:
            payload["stop"] = stop

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM API returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"LLM API request failed: {e}") from e

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e

    def health_check(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            url = f"{self.base_url}/models"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self._headers())
                return response.status_code == 200
        except Exception:
            return False
