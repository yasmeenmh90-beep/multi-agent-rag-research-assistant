"""
Router agent: looks at the question and the available domains, and decides
which domain retriever(s) should handle it. A question can legitimately
route to more than one domain.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, discover_domains
from app.agents.state import GraphState

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_parser = CommaSeparatedListOutputParser()

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a routing agent. Given a user question and a list of available "
     "knowledge domains, return only the domain name(s) most likely to contain "
     "the answer, as a comma-separated list. If unsure, return all domains. "
     "Available domains: {domains}\n{format_instructions}"),
    ("user", "{question}"),
])


def route(state: GraphState) -> GraphState:
    domains = discover_domains()
    if not domains:
        raise RuntimeError(
            "No ingested domains found. Run `python -m app.ingestion.ingest` first."
        )

    if len(domains) == 1:
        # Nothing to route between - skip the LLM call.
        return {**state, "domains": domains}

    chain = _ROUTER_PROMPT | _llm | _parser
    result = chain.invoke({
        "question": state["question"],
        "domains": ", ".join(domains),
        "format_instructions": _parser.get_format_instructions(),
    })

    picked = [d.strip() for d in result if d.strip() in domains]
    if not picked:
        picked = domains  # fail open rather than returning nothing

    return {**state, "domains": picked}
