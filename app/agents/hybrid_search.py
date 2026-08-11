"""
Hybrid retrieval: combines dense vector search (Chroma/embeddings) with
sparse keyword search (BM25) and merges the two ranked lists with
Reciprocal Rank Fusion (RRF).

Why: pure vector search misses exact terms/acronyms/model names that don't
embed distinctively (e.g. "GFM-RAG", "R^2AG"); pure keyword search misses
paraphrases and semantic matches. RRF gets the benefit of both without
needing to tune a blend weight.

BM25 index is built once per domain (from the same chunks already sitting
in Chroma) and cached in memory for the process lifetime.
"""
import re

from rank_bm25 import BM25Okapi

from app.config import RETRIEVER_TOP_K, RERANK_CANDIDATES
from app.vectorstore.store import get_vectorstore
from app.agents.reranker import rerank

_bm25_cache: dict[str, tuple] = {}  # domain -> (BM25Okapi | None, texts, metadatas)

RRF_K = 60  # standard RRF smoothing constant


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_bm25_index(domain: str):
    if domain not in _bm25_cache:
        store = get_vectorstore(domain)
        # Pull the raw chunks already embedded into Chroma so BM25 indexes
        # the exact same corpus the dense retriever sees - no separate
        # ingestion path to keep in sync.
        raw = store._collection.get(include=["documents", "metadatas"])
        texts = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None
        _bm25_cache[domain] = (bm25, texts, metadatas)

    return _bm25_cache[domain]


def _dense_search(domain: str, query: str, k: int) -> list[dict]:
    store = get_vectorstore(domain)
    docs = store.similarity_search(query, k=k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


def _sparse_search(domain: str, query: str, k: int) -> list[dict]:
    bm25, texts, metadatas = _get_bm25_index(domain)
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"content": texts[i], "metadata": metadatas[i]} for i in top_idx]


def _reciprocal_rank_fusion(result_lists: list[list[dict]], top_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    item_by_key: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            key = item["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            item_by_key[key] = item

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    return [item_by_key[k] for k in ranked_keys]


def hybrid_search(domain: str, query: str, k: int = RETRIEVER_TOP_K) -> list[dict]:
    """Dense + sparse search, fused via RRF, then cross-encoder reranked.

    We pull RERANK_CANDIDATES (wider than k) from the fused RRF results so
    the reranker has a real pool to work with, then keep only the top k
    after reranking.
    """
    candidate_k = max(k, RERANK_CANDIDATES)
    dense = _dense_search(domain, query, k=candidate_k)
    sparse = _sparse_search(domain, query, k=candidate_k)
    fused = _reciprocal_rank_fusion([dense, sparse], top_k=candidate_k)
    return rerank(query, fused, top_k=k)


def warm_bm25_cache(domains: list[str]) -> None:
    """Pre-build BM25 indexes for the given domains. Call this once at
    server startup so the first user query isn't stuck waiting for a
    large corpus to be pulled from Chroma and tokenized."""
    for domain in domains:
        _get_bm25_index(domain)
