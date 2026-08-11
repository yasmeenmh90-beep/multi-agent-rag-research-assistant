"""
Planner agent: decides whether the question is genuinely multi-part and, if
so, breaks it into focused sub-questions that get retrieved independently.
A simple question comes back unchanged as the only sub-question - this is
what makes the graph "agentic" rather than just adding a fixed extra hop for
every query.

Example: "Compare RAG hallucination mitigation vs multi-agent coordination
strategies" -> two sub-questions, each routed/retrieved on its own, then
merged before synthesis.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)


class QueryPlan(BaseModel):
    sub_questions: list[str] = Field(
        description=(
            "1 to 4 focused sub-questions that together cover the original "
            "question. If the question is already simple and single-topic, "
            "return it unchanged as the only item."
        )
    )


_parser = JsonOutputParser(pydantic_object=QueryPlan)

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query planning agent for a retrieval system. Decide whether "
     "the user's question is genuinely multi-part or comparative and would "
     "be better answered by decomposing it into separate focused "
     "sub-questions, each retrieved independently. "
     "If it is already a simple, single-topic question, do NOT decompose it "
     "- return it unchanged as the only sub-question.\n"
     "{format_instructions}"),
    ("user", "{question}"),
])


def plan(state: GraphState) -> GraphState:
    chain = _PLANNER_PROMPT | _llm | _parser
    result = chain.invoke({
        "question": state["question"],
        "format_instructions": _parser.get_format_instructions(),
    })

    sub_questions = result.get("sub_questions") or [state["question"]]
    sub_questions = [q.strip() for q in sub_questions if q.strip()][:4] or [state["question"]]

    return {
        **state,
        "sub_questions": sub_questions,
        "original_question": state["question"],
    }
