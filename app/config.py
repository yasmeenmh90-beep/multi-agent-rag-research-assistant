"""
Central configuration. Keep every tunable in one place so the rest of the
codebase never hardcodes a model name, chunk size, or domain list.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Models -----------------------------------------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# --- Vector store -------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

# --- Chunking -----------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ------------------------------------------------------------
RETRIEVER_TOP_K = 4

# --- Reranking ------------------------------------------------------------
# Hybrid search (dense + BM25 + RRF) pulls a wider net of candidates than
# we actually send to the synthesizer; the cross-encoder then re-scores
# that wider net and keeps only the top RETRIEVER_TOP_K. Pulling more
# candidates than the final k gives the reranker something to actually
# rerank instead of just re-confirming RRF's order.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 12

# --- Tracing (LangSmith) ----------------------------------------------
# LangChain/LangGraph pick up LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY /
# LANGCHAIN_PROJECT directly from the environment - no code wiring needed
# beyond setting them in .env. This flag is just for surfacing status
# (e.g. in /health) so it's visible whether tracing is actually active.
TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-rag")

# --- Domains --------------------------------------------------------------
# Auto-discovered from data/documents/<domain>/ subfolders at ingest time,
# but we keep a fallback list here so agents can reference domains before
# ingestion has ever run.
def discover_domains() -> list[str]:
    if not DOCUMENTS_DIR.exists():
        return []
    return sorted(
        p.name for p in DOCUMENTS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
