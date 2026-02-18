"""
Code retrieval system for RAG.

BASELINE: Keyword retrieval is provided and working.
TODO: Implement multi-strategy retrieval for Phase 1.
"""

import re
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

from .indexer import CodeChunk, CodebaseIndex

console = Console()


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""
    chunk: CodeChunk
    score: float
    method: str  # "keyword", "bm25", "embedding", "hybrid"


# ============================================================================
# BASELINE: Keyword retrieval (PROVIDED — do not modify)
# ============================================================================

def keyword_score(query: str, chunk: CodeChunk) -> float:
    """
    Simple keyword-based relevance scoring.

    Scores based on:
    - Term frequency in content
    - Symbol name matches (weighted higher)
    - File path relevance
    """
    query_terms = set(re.findall(r'\w+', query.lower()))
    if not query_terms:
        return 0.0

    score = 0.0
    content_lower = chunk.content.lower()

    # Term frequency scoring
    for term in query_terms:
        count = content_lower.count(term)
        if count > 0:
            score += min(count, 5) * 1.0  # Cap at 5 occurrences

    # Symbol name matching (higher weight)
    for symbol in chunk.symbols:
        symbol_lower = symbol.lower()
        for term in query_terms:
            if term in symbol_lower:
                score += 5.0  # Strong signal
            if symbol_lower == term:
                score += 10.0  # Exact match

    # File path relevance
    path_lower = chunk.file_path.lower()
    for term in query_terms:
        if term in path_lower:
            score += 3.0

    return score


def retrieve_keyword(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    file_filter: Optional[list[str]] = None,
) -> list[RetrievalResult]:
    """
    Retrieve top-k code chunks using keyword matching.

    This is the baseline retrieval method that works without
    any external dependencies.

    Args:
        index: The codebase index to search.
        query: Natural language query.
        top_k: Number of results to return.
        file_filter: Optional list of file paths to restrict search to.

    Returns:
        List of RetrievalResult sorted by relevance score.
    """
    results: list[RetrievalResult] = []

    for chunk in index.chunks:
        # Apply file filter if provided
        if file_filter:
            if not any(f in chunk.file_path for f in file_filter):
                continue

        score = keyword_score(query, chunk)
        if score > 0:
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                method="keyword",
            ))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ============================================================================
# BASELINE: Embedding retrieval stub (PROVIDED — candidates can improve)
# ============================================================================

def try_embedding_retrieval(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[list[RetrievalResult]]:
    """
    Attempt embedding-based retrieval using sentence-transformers.

    Returns None if sentence-transformers is not installed.
    Candidates can improve this function for better retrieval.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        console.print("[yellow]sentence-transformers not installed, falling back to keyword retrieval[/yellow]")
        return None

    try:
        model = SentenceTransformer(model_name)

        # Encode query
        query_embedding = model.encode(query, normalize_embeddings=True)

        # Encode all chunks (this is slow — candidates should optimize)
        chunk_texts = [
            f"File: {c.file_path}\nSymbols: {', '.join(c.symbols)}\n{c.content}"
            for c in index.chunks
        ]

        if not chunk_texts:
            return []

        chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)

        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                chunk=index.chunks[idx],
                score=float(similarities[idx]),
                method="embedding",
            ))

        return results

    except Exception as e:
        console.print(f"[yellow]Embedding retrieval failed: {e}[/yellow]")
        return None


# ============================================================================
# TODO: Implement multi-strategy retrieval
# ============================================================================

def retrieve(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    file_filter: Optional[list[str]] = None,
    use_embeddings: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    strategy: str = "keyword",
) -> list[RetrievalResult]:
    """
    Main retrieval function. Supports multiple strategies.

    Currently only "keyword" strategy works.
    TODO: Implement "bm25", "hybrid", and "embedding" strategies for Phase 1.
    """
    if strategy == "bm25":
        # TODO: Implement BM25 retrieval path
        console.print("[yellow]BM25 strategy not implemented, falling back to keyword[/yellow]")
        return retrieve_keyword(index, query, top_k, file_filter)

    elif strategy == "hybrid":
        # TODO: Implement hybrid retrieval path
        console.print("[yellow]Hybrid strategy not implemented, falling back to keyword[/yellow]")
        return retrieve_keyword(index, query, top_k, file_filter)

    elif strategy == "embedding":
        if use_embeddings:
            results = try_embedding_retrieval(index, query, top_k, embedding_model)
            if results is not None:
                return results
        return retrieve_keyword(index, query, top_k, file_filter)

    else:  # keyword (default)
        return retrieve_keyword(index, query, top_k, file_filter)


# ============================================================================
# Context formatting (PROVIDED — do not modify)
# ============================================================================

def format_context(results: list[RetrievalResult], max_tokens: int = 4000) -> str:
    """
    Format retrieval results into a context string for the LLM.

    Args:
        results: Retrieved code chunks.
        max_tokens: Approximate max character budget (rough estimate).

    Returns:
        Formatted context string.
    """
    parts: list[str] = []
    total_len = 0

    for r in results:
        header = f"=== {r.chunk.file_path} (lines {r.chunk.start_line}-{r.chunk.end_line}, score: {r.score:.2f}) ==="
        section = f"{header}\n{r.chunk.content}\n"

        if total_len + len(section) > max_tokens * 4:  # rough char-to-token ratio
            break

        parts.append(section)
        total_len += len(section)

    return "\n".join(parts)
