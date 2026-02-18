"""
BM25 (Okapi BM25) retrieval implementation.

Provides tokenization with camelCase splitting, IDF computation,
BM25 index construction, and retrieval over CodeChunk objects.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .indexer import CodeChunk, CodebaseIndex


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can",
        "could", "of", "in", "to", "for", "with", "on", "at", "from",
        "by", "about", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "this", "that", "these", "those", "it", "its",
    }
)


def tokenize(text: str) -> list[str]:
    """Tokenize text with camelCase splitting and stop-word removal.

    Steps:
      1. Split on non-alphanumeric characters.
      2. Further split camelCase tokens (e.g. ``togglePause`` → ``toggle``, ``Pause``).
      3. Lower-case everything and remove stop words / single-char tokens.
    """
    raw_tokens = re.findall(r"[A-Za-z0-9]+", text)
    tokens: list[str] = []
    for tok in raw_tokens:
        parts = _CAMEL_RE.split(tok)
        for part in parts:
            low = part.lower()
            if len(low) > 1 and low not in _STOP_WORDS:
                tokens.append(low)
    return tokens


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    """Pre-computed BM25 index over a set of documents (code chunks)."""

    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avg_dl: float = 0.0
    n_docs: int = 0
    # term → doc_id set
    df: dict[str, int] = field(default_factory=dict)
    # Inverse Document Frequency cache: term → idf value
    idf_cache: dict[str, float] = field(default_factory=dict)
    # term → {doc_id: term_freq}
    tf: dict[str, dict[int, int]] = field(default_factory=dict)


def _compute_idf(df: int, n_docs: int) -> float:
    """Compute IDF using the standard Okapi BM25 formula.

    idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    """
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def build_bm25_index(chunks: list[CodeChunk]) -> BM25Index:
    """Build a BM25 index from a list of code chunks.

    Each chunk's ``content`` is tokenized and indexed for fast scoring.
    """
    index = BM25Index()
    index.n_docs = len(chunks)

    total_len = 0
    for doc_id, chunk in enumerate(chunks):
        # Combine file path, symbols, and content for richer matching
        text = f"{chunk.file_path} {' '.join(chunk.symbols)} {chunk.content}"
        tokens = tokenize(text)
        index.doc_tokens.append(tokens)
        index.doc_lengths.append(len(tokens))
        total_len += len(tokens)

        # Count term frequencies
        seen: set[str] = set()
        for tok in tokens:
            # TF
            if tok not in index.tf:
                index.tf[tok] = {}
            index.tf[tok][doc_id] = index.tf[tok].get(doc_id, 0) + 1
            # DF (count each term once per document)
            if tok not in seen:
                index.df[tok] = index.df.get(tok, 0) + 1
                seen.add(tok)

    index.avg_dl = total_len / max(index.n_docs, 1)

    # Pre-compute IDF values
    for term, df_val in index.df.items():
        index.idf_cache[term] = _compute_idf(df_val, index.n_docs)

    return index


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def bm25_score(
    query_tokens: list[str],
    doc_id: int,
    bm25_index: BM25Index,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute the BM25 score for a single document given query tokens.

    Uses Okapi BM25 with parameters *k1* and *b*.
    """
    score = 0.0
    dl = bm25_index.doc_lengths[doc_id]
    avg_dl = bm25_index.avg_dl

    for term in query_tokens:
        if term not in bm25_index.idf_cache:
            continue

        idf = bm25_index.idf_cache[term]
        tf_val = bm25_index.tf.get(term, {}).get(doc_id, 0)

        # BM25 TF component
        numerator = tf_val * (k1 + 1.0)
        denominator = tf_val + k1 * (1.0 - b + b * (dl / max(avg_dl, 1e-9)))
        score += idf * (numerator / denominator)

    return score


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_bm25(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    file_filter: Optional[list[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[CodeChunk, float]]:
    """Retrieve the top-k code chunks for *query* using BM25 scoring.

    Returns a list of ``(chunk, score)`` tuples sorted by descending score.
    """
    if not index.chunks:
        return []

    # Apply file filter first to reduce index size
    if file_filter:
        filtered_chunks = [
            c for c in index.chunks
            if any(f in c.file_path for f in file_filter)
        ]
    else:
        filtered_chunks = list(index.chunks)

    if not filtered_chunks:
        return []

    bm25_idx = build_bm25_index(filtered_chunks)
    query_tokens = tokenize(query)

    scored: list[tuple[int, float]] = []
    for doc_id in range(bm25_idx.n_docs):
        s = bm25_score(query_tokens, doc_id, bm25_idx, k1=k1, b=b)
        if s > 0:
            scored.append((doc_id, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[tuple[CodeChunk, float]] = []
    for doc_id, s in scored[:top_k]:
        results.append((filtered_chunks[doc_id], s))

    return results
