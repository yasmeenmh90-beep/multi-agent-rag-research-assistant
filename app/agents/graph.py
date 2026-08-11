"""
Wires contextualizer -> router -> planner -> retriever -> synthesizer ->
critic into a LangGraph StateGraph, with a smart retry loop: if the critic
finds the draft answer isn't grounded, a rewriter agent reformulates the
question (using the critic's feedback) before retrieving again.

    contextualizer -> router -> planner -> retriever -> synthesizer -> critic --grounded--> finalize
                                                ^                           |
                                                └──────── rewriter (max 1 retry) ───────────┘

- contextualizer: rewrites follow-up questions into standalone questions
  using chat history, so multi-turn conversation works
- planner: decomposes multi-part questions into focused sub-questions
- retriever: hybrid (dense + BM25) search per sub-question per domain
- rewriter: reformulates the question using critic feedback on retry
"""
from langgraph.graph import StateGraph, END

from app.agents.state import GraphState
from app.agents.contextualizer import contextualize
from app.agents.router import route
from app.agents.planner import plan
from app.agents.retriever import retrieve
from app.agents.synthesizer import synthesize
from app.agents.critic import critique, should_retry
from app.agents.rewriter import rewrite
from app.agents.finalize import finalize


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("contextualizer", contextualize)
    graph.add_node("router", route)
    graph.add_node("planner", plan)
    graph.add_node("retriever", retrieve)
    graph.add_node("synthesizer", synthesize)
    graph.add_node("critic", critique)
    graph.add_node("rewriter", rewrite)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("contextualizer")
    graph.add_edge("contextualizer", "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "rewriter", "finalize": "finalize"},
    )
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import time; reused across requests.
app_graph = build_graph()


def run_query(
    question: str,
    chat_history: list[dict] | None = None,
    session_id: str | None = None,
    explain_simply: bool = False,
) -> GraphState:
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "retry_count": 0,
        "explain_simply": explain_simply,
    }

    # run_name/tags/metadata don't change behavior - they just make this
    # run readable and filterable in the LangSmith trace dashboard when
    # LANGCHAIN_TRACING_V2=true is set (see .env.example).
    trace_config = {
        "run_name": "multi_agent_rag_query",
        "tags": ["multi-agent-rag"],
        "metadata": {"session_id": session_id} if session_id else {},
    }

    return app_graph.invoke(initial_state, config=trace_config)