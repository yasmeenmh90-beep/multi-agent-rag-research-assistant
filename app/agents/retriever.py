"""
Retriever agent: for every sub-question the planner produced and every
domain the router picked, run hybrid (dense + sparse) search and merge the
results, de-duplicating chunks that show up more than once (a chunk can
legitimately match multiple sub-questions).
"""
from app.config import RETRIEVER_TOP_K
from app.agents.hybrid_search import hybrid_search
from app.agents.state import GraphState


def retrieve(state: GraphState) -> GraphState:
    sub_questions = state.get("sub_questions") or [state["question"]]

    all_hits = []
    seen_keys = set()

    for sub_q in sub_questions:
        for domain in state["domains"]:
            hits = hybrid_search(domain, sub_q, k=RETRIEVER_TOP_K)
            for h in hits:
                # de-dupe on domain + a content prefix (cheap, good enough
                # since chunks are unlikely to share a 120-char prefix)
                key = (domain, h["content"][:120])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_hits.append({
                    "domain": domain,
                    "source": h["metadata"].get("source", "unknown"),
                    "content": h["content"],
                })

    return {**state, "retrieved": all_hits}
