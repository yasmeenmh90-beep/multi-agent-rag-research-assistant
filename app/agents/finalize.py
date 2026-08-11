"""
Final node: packages the draft answer into the final response. If the critic
never confirmed grounding (e.g. retries exhausted), the answer is prefixed
with an explicit caveat instead of silently presenting it as verified.
"""
from app.agents.state import GraphState


def finalize(state: GraphState) -> GraphState:
    answer = state["draft_answer"]
    if not state.get("is_grounded"):
        answer = (
            "Note: I could not fully verify this answer against the retrieved "
            "sources. Treat it as provisional.\n\n" + answer
        )
    return {**state, "final_answer": answer}
