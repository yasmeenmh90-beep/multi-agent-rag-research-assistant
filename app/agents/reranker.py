"""
Cross-encoder reranker: hybrid search (dense + BM25 + RRF) is good at
recall - it casts a wide net of plausibly-relevant chunks - but RRF's rank
fusion is a cheap positional heuristic, not a real relevance judgment made
by actually reading the query against the passage together.

A cross-encoder scores each (query, passage) pair jointly through a small
transformer - much more accurate than comparing embeddings or keyword
overlap independently, because the model attends across both texts at
once instead of comparing two separately-computed vectors. This trades a
bit of latency for noticeably better precision on the final top-k that
actually reaches the synthesizer.
"""
from sentence_transformers import CrossEncoder

from app.config import RERANK_MODEL

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def warm_reranker() -> None:
    """Load the cross-encoder model at process startup rather than on the
    first request - the model download/load (~90MB, one-time, cached under
    ~/.cache/huggingface after that) would otherwise stall the first query."""
    _get_model()


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score `candidates` (each a {"content", "metadata"} dict) against
    `query` with the cross-encoder and return the top_k, best first."""
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [item for item, _ in scored[:top_k]]
