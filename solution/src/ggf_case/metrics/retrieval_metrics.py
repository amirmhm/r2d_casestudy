"""
Retrieval quality metrics: Precision@k, Recall@k, MRR, NDCG@k, Hit Rate.

All functions accept lists of retrieved items and relevant (ground-truth) items.
``compute_retrieval_scores`` aggregates over multiple queries.
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalScores:
    """Aggregated retrieval quality scores."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    num_queries: int = 0


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved items."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant item.

    Returns 0.0 if no relevant item is found.
    """
    relevant_set = set(relevant)
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Uses binary relevance: 1 if the item is in ``relevant``, else 0.
    """
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG: all relevant items at the top positions
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved: list[str], relevant: list[str]) -> float:
    """Returns 1.0 if at least one relevant item is in retrieved, else 0.0."""
    relevant_set = set(relevant)
    for item in retrieved:
        if item in relevant_set:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_retrieval_scores(
    queries: list[dict],
    k: int = 5,
) -> RetrievalScores:
    """Compute averaged retrieval scores over multiple queries.

    Each element of *queries* should be a dict with keys:
      - ``retrieved``: list of retrieved file paths
      - ``relevant``: list of ground-truth relevant file paths

    Returns a populated :class:`RetrievalScores` dataclass.
    """
    if not queries:
        return RetrievalScores()

    total_p = 0.0
    total_r = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_hit = 0.0
    n = len(queries)

    for q in queries:
        retrieved = q.get("retrieved", [])
        relevant = q.get("relevant", [])

        total_p += precision_at_k(retrieved, relevant, k)
        total_r += recall_at_k(retrieved, relevant, k)
        total_mrr += mrr(retrieved, relevant)
        total_ndcg += ndcg_at_k(retrieved, relevant, k)
        total_hit += hit_rate(retrieved, relevant)

    return RetrievalScores(
        precision_at_k=total_p / n,
        recall_at_k=total_r / n,
        mrr=total_mrr / n,
        ndcg_at_k=total_ndcg / n,
        hit_rate=total_hit / n,
        num_queries=n,
    )
