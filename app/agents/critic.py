"""
Critic agent: checks whether the draft answer is actually supported by the
retrieved chunks. Returns structured JSON (is_grounded + feedback) so the
graph can branch on it - this is the piece that turns a plain RAG chain
into something that can catch and correct its own hallucinations.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)


class CriticVerdict(BaseModel):
    is_grounded: bool = Field(
        description="True only if every claim in the answer is directly supported by the context"
    )
    feedback: str = Field(
        description="If not grounded, a short note on what's unsupported or missing"
    )


_parser = JsonOutputParser(pydantic_object=CriticVerdict)

_CRITIC_PROMPT = PromptTemplate(
    template=(
        "You are a strict fact-checking agent. Given the context and a draft "
        "answer, decide if the answer is fully supported by the context.\n\n"
        "Context:\n{context}\n\n"
        "Draft answer:\n{draft_answer}\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["context", "draft_answer"],
    partial_variables={"format_instructions": _parser.get_format_instructions()},
)

MAX_RETRIES = 1


def _format_context(retrieved: list[dict]) -> str:
    return "\n\n".join(r["content"] for r in retrieved)


def critique(state: GraphState) -> GraphState:
    chain = _CRITIC_PROMPT | _llm | _parser
    verdict = chain.invoke({
        "context": _format_context(state.get("retrieved", [])),
        "draft_answer": state["draft_answer"],
    })

    return {
        **state,
        "is_grounded": bool(verdict.get("is_grounded")),
        "critic_feedback": verdict.get("feedback"),
        "retry_count": state.get("retry_count", 0),
    }


def should_retry(state: GraphState) -> str:
    """Conditional edge: retry retrieval once if ungrounded, otherwise finish."""
    if state.get("is_grounded"):
        return "finalize"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "finalize"  # give up gracefully rather than loop forever
    return "retry"


def bump_retry(state: GraphState) -> GraphState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}
