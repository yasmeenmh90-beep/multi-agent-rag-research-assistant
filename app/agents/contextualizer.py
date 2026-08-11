"""
Contextualizer agent: the first node in the graph. If there's prior chat
history, it rewrites the incoming question into a standalone question that
carries whatever context a follow-up implicitly depends on - e.g. "what
about its limitations?" becomes "what are the limitations of hybrid
retrieval in RAG systems?" using the previous turn.

This is what makes multi-turn conversation work without threading history
through every downstream agent: everything after this node (router, planner,
retrievers, synthesizer, critic) only ever sees a self-contained question,
so none of that logic needs to change.

If there's no history yet (first turn), this is a no-op - skips the LLM
call entirely rather than paying latency for nothing.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, rewrite the "
     "follow-up into a standalone question that contains all context needed "
     "to understand it without the history. If the follow-up question is "
     "already standalone (doesn't depend on prior context), return it "
     "unchanged. Return ONLY the question - no preamble, no explanation.\n\n"
     "Conversation history:\n{history}"),
    ("user", "{question}"),
])


def _format_history(chat_history: list[dict]) -> str:
    lines = []
    for turn in chat_history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def contextualize(state: GraphState) -> GraphState:
    raw_question = state["question"]
    chat_history = state.get("chat_history") or []

    if not chat_history:
        # First turn - nothing to contextualize against.
        return {**state, "raw_question": raw_question}

    chain = _CONTEXTUALIZE_PROMPT | _llm | StrOutputParser()
    standalone = chain.invoke({
        "question": raw_question,
        "history": _format_history(chat_history),
    }).strip()

    return {**state, "question": standalone, "raw_question": raw_question}
