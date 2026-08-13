"""
Generates a structured literature review for a topic, using the papers
already ingested by fetch_and_ingest_topic.py into that topic's domain.

Unlike the main Q&A synthesizer (one question -> one answer), this pulls a
broad set of chunks across several angles on the topic (overview, methods,
findings, limitations) and asks the LLM to write a proper review: themed
sections, each claim cited by source filename, plus an explicit gaps
section. Metadata-only papers (found but not ingested - no open-access PDF)
are listed separately so the review doesn't pretend to have read them.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.agents.hybrid_search import hybrid_search

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

# Broad angles to pull chunks from, rather than one narrow query - a real
# review needs coverage of what's been done, how, and what's missing, not
# just a direct answer to a single question.
_REVIEW_ANGLES = [
    "overview and background of {topic}",
    "methods and approaches used for {topic}",
    "key findings and results related to {topic}",
    "limitations, open problems, or gaps in research on {topic}",
]

_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are writing a literature review section for a student's thesis on: {topic}\n\n"
     "You are given excerpts from papers that were actually retrieved for this topic. "
     "Write a structured literature review with these sections:\n"
     "1. Overview - 2-3 sentences framing the topic\n"
     "2. Key Themes - organize the findings into 2-4 themes, each a short paragraph\n"
     "3. Research Gaps - what's underexplored or conflicting, based only on what's in the excerpts\n\n"
     "Rules:\n"
     "- Cite every specific claim with its source filename in square brackets, e.g. [paper_name.pdf]\n"
     "- Only use the provided excerpts - do not add outside knowledge or invent findings\n"
     "- If the excerpts don't support a strong gaps section, say the gaps are unclear from "
     "the retrieved material rather than guessing\n"
     "- Write in formal academic prose suitable for pasting into a thesis draft\n\n"
     "Excerpts:\n{context}"),
    ("user", "Write the literature review."),
])


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['source']}]\n{c['content']}" for c in chunks
    )


def generate_literature_review(topic: str, domain: str, top_k_per_angle: int = 6) -> dict:
    """Pulls chunks across several angles on the topic, dedupes, and
    synthesizes a structured review. Returns {"review_text": str,
    "sources_used": list[str]}."""
    seen_keys = set()
    all_chunks = []

    for angle_template in _REVIEW_ANGLES:
        query = angle_template.format(topic=topic)
        hits = hybrid_search(domain, query, k=top_k_per_angle)
        for h in hits:
            key = h["content"][:120]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_chunks.append({
                "source": h["metadata"].get("source", "unknown"),
                "content": h["content"],
            })

    if not all_chunks:
        return {
            "review_text": (
                "No content could be retrieved for this topic - either no papers were "
                "successfully ingested, or the ingested papers don't contain relevant text."
            ),
            "sources_used": [],
        }

    context = _format_context(all_chunks)
    chain = _REVIEW_PROMPT | _llm | StrOutputParser()
    review_text = chain.invoke({"topic": topic, "context": context})

    sources_used = sorted({c["source"] for c in all_chunks})
    return {"review_text": review_text, "sources_used": sources_used}

