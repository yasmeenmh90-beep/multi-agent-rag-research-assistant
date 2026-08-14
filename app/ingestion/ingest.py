"""
Walks data/documents/<domain>/*, loads each file, chunks it, and writes the
chunks into that domain's Chroma collection.

Run directly:
    python -m app.ingestion.ingest
"""
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, discover_domains
from app.vectorstore.store import add_documents

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_file(path: Path):
    loader_cls = LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        print(f"  skipping unsupported file type: {path.name}")
        return []
    try:
        return loader_cls(str(path)).load()
    except Exception as exc:  # noqa: BLE001 - ingestion should keep going
        print(f"  failed to load {path.name}: {exc}")
        return []


def ingest_domain(domain: str, splitter: RecursiveCharacterTextSplitter) -> tuple[int, int]:
    domain_dir = DOCUMENTS_DIR / domain
    files = [p for p in domain_dir.iterdir() if p.is_file()]

    all_docs = []
    for f in files:
        docs = load_file(f)
        for d in docs:
            d.metadata["source"] = f.name
            d.metadata["domain"] = domain
        all_docs.extend(docs)

    chunks = splitter.split_documents(all_docs)
    # Malformed PDFs (the "Ignoring wrong pointing object" warnings during
    # loading are the tell) can produce chunks with blank or whitespace-only
    # text. Chroma will happily store these, but on retrieval an empty
    # string can come back as None, which crashes langchain's Document
    # model ("none is not an allowed value") the moment that chunk is
    # retrieved - so these are filtered out here, before they're ever
    # embedded, rather than trying to handle it at query time.
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if chunks:
        add_documents(domain, chunks)

    return len(files), len(chunks)


def main():
    domains = discover_domains()
    if not domains:
        print(
            f"No domain folders found under {DOCUMENTS_DIR}.\n"
            "Create subfolders like data/documents/research_papers/ and add files."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_files, total_chunks = 0, 0
    print(f"Discovered domains: {domains}\n")
    for domain in domains:
        n_files, n_chunks = ingest_domain(domain, splitter)
        total_files += n_files
        total_chunks += n_chunks
        print(f"[{domain}] {n_files} files -> {n_chunks} chunks")

    print(f"\nDone. {total_files} total documents, {total_chunks} total chunks "
          f"across {len(domains)} domain(s).")


if __name__ == "__main__":
    main()
