"""
Rewriter agent: when the critic finds the draft answer isn't grounded, this
agent reformulates the question - using the critic's own feedback about
what was missing - instead of blindly re-running the exact same query
against the exact same index. This is what makes the retry loop worth
having; retrying an unchanged query against an unchanged index would just
reproduce the same miss.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Retrieval did not find enough evidence to ground an answer to the "
     "user's question. Critic feedback on what was missing: {feedback}\n\n"
     "Rewrite the question to use more specific terminology, synonyms, or a "
     "narrower focus that is more likely to match relevant passages in a "
     "technical document corpus. Return ONLY the rewritten question - no "
     "preamble, no explanation."),
    ("user", "{question}"),
])


def rewrite(state: GraphState) -> GraphState:
    chain = _REWRITE_PROMPT | _llm | StrOutputParser()

    new_question = chain.invoke({
        "question": state.get("original_question", state["question"]),
        "feedback": state.get("critic_feedback") or "No specific feedback provided.",
    }).strip()

    return {
        **state,
        "question": new_question,
        "sub_questions": [new_question],
        "retry_count": state.get("retry_count", 0) + 1,
    }
