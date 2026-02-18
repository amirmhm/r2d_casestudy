"""
Reranking module for retrieved code chunks.

Provides an abstract ``Reranker`` base class, a ``NoOpReranker`` (pass-through),
and an optional ``CrossEncoderReranker`` that uses sentence-transformers.
Use ``create_reranker()`` factory to instantiate the appropriate implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional

from rich.console import Console

from .indexer import CodeChunk

console = Console()


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[tuple[CodeChunk, float]],
        top_k: int = 8,
    ) -> list[tuple[CodeChunk, float]]:
        """Rerank a list of (chunk, score) pairs and return top-k."""
        ...


class NoOpReranker(Reranker):
    """Pass-through reranker that returns results unchanged."""

    def rerank(
        self,
        query: str,
        chunks: list[tuple[CodeChunk, float]],
        top_k: int = 8,
    ) -> list[tuple[CodeChunk, float]]:
        return chunks[:top_k]


class CrossEncoderReranker(Reranker):
    """Reranker based on a cross-encoder model from sentence-transformers.

    Falls back to ``NoOpReranker`` behaviour if sentence-transformers
    is not installed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except ImportError:
            console.print(
                "[yellow]sentence-transformers not installed — "
                "CrossEncoderReranker will act as NoOp[/yellow]"
            )
        except Exception as exc:
            console.print(f"[yellow]Failed to load cross-encoder model: {exc}[/yellow]")

    def rerank(
        self,
        query: str,
        chunks: list[tuple[CodeChunk, float]],
        top_k: int = 8,
    ) -> list[tuple[CodeChunk, float]]:
        if self._model is None or not chunks:
            return chunks[:top_k]

        pairs = [
            (query, f"{c.file_path}\n{' '.join(c.symbols)}\n{c.content}")
            for c, _score in chunks
        ]

        try:
            scores = self._model.predict(
                [(p[0], p[1]) for p in pairs]
            )
            reranked = [
                (chunks[i][0], float(scores[i]))
                for i in range(len(chunks))
            ]
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]
        except Exception as exc:
            console.print(f"[yellow]Cross-encoder reranking failed: {exc}[/yellow]")
            return chunks[:top_k]


def create_reranker(
    enabled: bool = False,
    model_name: Optional[str] = None,
) -> Reranker:
    """Factory function to create the appropriate reranker.

    Args:
        enabled: If ``True``, attempt to create a ``CrossEncoderReranker``.
        model_name: Optional cross-encoder model name.

    Returns:
        A ``Reranker`` instance.
    """
    if not enabled:
        return NoOpReranker()

    model = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return CrossEncoderReranker(model_name=model)
