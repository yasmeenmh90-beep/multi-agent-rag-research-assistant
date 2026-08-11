"""
Shared state object that flows through every node in the LangGraph.
Keeping this explicit (rather than a loose dict) makes the graph easy to
reason about and easy to explain in an interview.
"""
from typing import TypedDict, Optional


class GraphState(TypedDict, total=False):
    question: str
    raw_question: str
    chat_history: list[dict]
    original_question: str
    sub_questions: list[str]
    domains: list[str]          # which domain(s) the router picked
    retrieved: list[dict]       # [{domain, source, content}]
    draft_answer: str
    is_grounded: bool
    critic_feedback: Optional[str]
    retry_count: int
    final_answer: str
    sources: list[str]
    explain_simply: bool        # if True, synthesizer answers in
                                 # beginner-friendly plain language