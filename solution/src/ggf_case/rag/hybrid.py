"""
Hybrid retrieval: combines keyword, BM25, and embedding retrieval results
using Reciprocal Rank Fusion (RRF) or weighted combination.
"""

from typing import Optional

from rich.console import Console

from .indexer import CodeChunk, CodebaseIndex
from .retriever import RetrievalResult, retrieve_keyword, try_embedding_retrieval
from .bm25 import retrieve_bm25

console = Console()


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a document *d*:

        score(d) = sum over lists  1 / (k + rank_i(d))

    where *k* is a constant (default 60, per the original RRF paper).
    """
    # chunk key → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk key → best RetrievalResult (to preserve metadata)
    best_result: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            key = f"{result.chunk.file_path}:{result.chunk.start_line}"
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
            if key not in best_result or result.score > best_result[key].score:
                best_result[key] = result

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused: list[RetrievalResult] = []
    for key in sorted_keys:
        original = best_result[key]
        fused.append(
            RetrievalResult(
                chunk=original.chunk,
                score=rrf_scores[key],
                method="hybrid",
            )
        )

    return fused


def weighted_combination(
    result_lists: list[list[RetrievalResult]],
    weights: Optional[list[float]] = None,
) -> list[RetrievalResult]:
    """Combine multiple result lists using weighted score summation.

    Each result list's scores are min-max normalised before combining.
    Default weights are equal (1.0 per list).
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    combined: dict[str, float] = {}
    best_result: dict[str, RetrievalResult] = {}

    for result_list, weight in zip(result_lists, weights):
        if not result_list:
            continue

        # Min-max normalisation
        scores = [r.score for r in result_list]
        min_s = min(scores)
        max_s = max(scores)
        score_range = max_s - min_s if max_s != min_s else 1.0

        for r in result_list:
            key = f"{r.chunk.file_path}:{r.chunk.start_line}"
            norm_score = (r.score - min_s) / score_range
            combined[key] = combined.get(key, 0.0) + weight * norm_score
            if key not in best_result:
                best_result[key] = r

    sorted_keys = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

    results: list[RetrievalResult] = []
    for key in sorted_keys:
        original = best_result[key]
        results.append(
            RetrievalResult(
                chunk=original.chunk,
                score=combined[key],
                method="hybrid",
            )
        )

    return results


def hybrid_retrieve(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    file_filter: Optional[list[str]] = None,
    use_embeddings: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    fusion_method: str = "rrf",
) -> list[RetrievalResult]:
    """Combine keyword, BM25, and (optionally) embedding retrieval.

    Args:
        index: The codebase index.
        query: Natural language query.
        top_k: Number of results to return.
        file_filter: Optional file path filter.
        use_embeddings: Whether to include embedding results.
        embedding_model: Name of the sentence-transformers model.
        fusion_method: ``"rrf"`` (Reciprocal Rank Fusion) or ``"weighted"``.

    Returns:
        Fused list of ``RetrievalResult``.
    """
    # Gather candidate lists — fetch more than top_k from each source
    fetch_k = top_k * 3

    # 1. Keyword retrieval
    keyword_results = retrieve_keyword(index, query, top_k=fetch_k, file_filter=file_filter)

    # 2. BM25 retrieval
    bm25_raw = retrieve_bm25(index, query, top_k=fetch_k, file_filter=file_filter)
    bm25_results = [
        RetrievalResult(chunk=chunk, score=score, method="bm25")
        for chunk, score in bm25_raw
    ]

    result_lists: list[list[RetrievalResult]] = [keyword_results, bm25_results]

    # 3. Embedding retrieval (optional)
    if use_embeddings:
        embedding_results = try_embedding_retrieval(index, query, top_k=fetch_k, model_name=embedding_model)
        if embedding_results:
            result_lists.append(embedding_results)

    # Fuse
    if fusion_method == "weighted":
        fused = weighted_combination(result_lists)
    else:
        fused = reciprocal_rank_fusion(result_lists, k=60)

    return fused[:top_k]
