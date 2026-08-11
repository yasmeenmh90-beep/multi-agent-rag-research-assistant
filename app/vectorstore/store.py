"""
Thin wrapper around Chroma that keeps one collection per document domain.
This is what makes "one retriever agent per domain" possible: each domain
gets its own isolated collection instead of everything living in one big
index with a metadata filter bolted on.
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, OPENAI_API_KEY

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

_collection_cache: dict[str, Chroma] = {}


def get_vectorstore(domain: str) -> Chroma:
    """Return (and cache) the Chroma collection for a given domain."""
    if domain not in _collection_cache:
        _collection_cache[domain] = Chroma(
            collection_name=f"domain_{domain}",
            embedding_function=_embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _collection_cache[domain]


def add_documents(domain: str, documents: list) -> int:
    """Embed and persist a batch of chunked Documents into a domain's collection."""
    store = get_vectorstore(domain)
    store.add_documents(documents)
    return len(documents)
