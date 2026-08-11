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
from app.config import RERANK_MODEL, ENABLE_RERANKER

_model = None


def _get_model():
    # Imported lazily, only when reranking is actually enabled - this is
    # what keeps sentence_transformers/torch out of the process entirely
    # on memory-capped deploys where ENABLE_RERANKER is left off.
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def warm_reranker() -> None:
    """Load the cross-encoder model at process startup rather than on the
    first request - the model download/load (~90MB, one-time, cached under
    ~/.cache/huggingface after that) would otherwise stall the first query.
    No-op when ENABLE_RERANKER is off."""
    if ENABLE_RERANKER:
        _get_model()


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score `candidates` (each a {"content", "metadata"} dict) against
    `query` with the cross-encoder and return the top_k, best first.

    If ENABLE_RERANKER is off, just truncates the already RRF-fused list
    to top_k without loading the cross-encoder - lower precision on the
    final results, but avoids torch/sentence-transformers entirely."""
    if not candidates:
        return []

    if not ENABLE_RERANKER:
        return candidates[:top_k]

    model = _get_model()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [item for item, _ in scored[:top_k]]