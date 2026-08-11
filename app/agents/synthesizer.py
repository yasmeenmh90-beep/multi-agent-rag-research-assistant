"""
Synthesizer agent: takes the retrieved chunks and drafts an answer that
cites its sources. This is deliberately separate from the critic - one
agent writes, another checks - which mirrors how you'd want this to work
in production (never trust a single LLM call to both generate and grade
itself well).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

_BASE_RULES = (
    "Answer the user's question using ONLY the provided context chunks. "
    "Cite each fact with its source filename in square brackets, e.g. [report.pdf]. "
    "If the context does not contain the answer, say so explicitly rather than guessing.\n\n"
)

_SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _BASE_RULES + "Context:\n{context}"),
    ("user", "{question}"),
])

# "Explain simply" mode: same grounding/citation rules, but written for a
# beginner who may be new to the topic - plain language, no unexplained
# jargon, short sentences. Still cites sources and still refuses to guess.
_SYNTH_PROMPT_SIMPLE = ChatPromptTemplate.from_messages([
    ("system",
     _BASE_RULES +
     "Additionally: explain this in simple, beginner-friendly language, as if "
     "to someone new to the field. Avoid unexplained jargon - if you must use "
     "a technical term, briefly define it in plain words the first time you "
     "use it. Prefer short sentences and concrete examples over dense "
     "academic phrasing. Do not oversimplify to the point of being "
     "inaccurate - the citation and grounding rules above still apply.\n\n"
     "Context:\n{context}"),
    ("user", "{question}"),
])


def _format_context(retrieved: list[dict]) -> str:
    return "\n\n".join(
        f"[{r['source']}] ({r['domain']})\n{r['content']}" for r in retrieved
    )


def synthesize(state: GraphState) -> GraphState:
    context = _format_context(state.get("retrieved", []))
    prompt = _SYNTH_PROMPT_SIMPLE if state.get("explain_simply") else _SYNTH_PROMPT
    chain = prompt | _llm | StrOutputParser()

    draft = chain.invoke({
        "question": state["question"],
        "context": context if context else "No context retrieved.",
    })

    sources = sorted({r["source"] for r in state.get("retrieved", [])})
    return {**state, "draft_answer": draft, "sources": sources}