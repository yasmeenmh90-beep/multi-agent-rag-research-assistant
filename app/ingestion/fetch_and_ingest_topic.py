"""
Given a topic, searches arXiv + Semantic Scholar, downloads whatever PDFs
are actually available (many Semantic Scholar results won't have one - not
every paper is open-access), and ingests them into a domain named after the
topic so the existing retrieval pipeline can be used unchanged.

Papers without a downloadable PDF are still kept in the returned metadata
list (for the bibliography) even though their content isn't in the corpus -
the literature review synthesizer is told which papers actually got ingested
vs. metadata-only, so it doesn't claim to have read something it hasn't.
"""
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.paper_search import search_topic
from app.ingestion.ingest import ingest_domain
from app.agents.hybrid_search import warm_bm25_cache

REQUEST_DELAY_SEC = 1


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")[:60]


def _download_pdf(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  failed to download {url}: {exc}")
        return False


def fetch_and_ingest_topic(topic: str, max_per_source: int = 15) -> dict:
    """Returns:
    {
        "domain": str,
        "papers": list[dict],       # all found, with "ingested": bool per paper
        "num_found": int,
        "num_ingested": int,
        "num_chunks": int,
    }
    """
    # A timestamp suffix guarantees a fresh domain every run - without it,
    # re-searching the same topic would reuse the old domain folder and
    # ingest_domain() would pick up every PDF ever downloaded for that slug,
    # so the generated review could cite papers that aren't in this run's
    # "papers found" list at all (confusing, and technically inaccurate -
    # the review would claim to summarize a set of papers it wasn't shown).
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    domain = f"lit_{slugify(topic)}_{run_id}"
    domain_dir = DOCUMENTS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    papers = search_topic(topic, max_per_source=max_per_source)

    for paper in papers:
        paper["ingested"] = False
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            continue  # metadata-only, e.g. a paywalled Semantic Scholar result

        filename = f"{slugify(paper['title'])}.pdf"
        dest = domain_dir / filename

        if dest.exists():
            paper["ingested"] = True
            paper["filename"] = filename
            continue

        ok = _download_pdf(pdf_url, dest)
        if ok:
            paper["ingested"] = True
            paper["filename"] = filename
        time.sleep(REQUEST_DELAY_SEC)

    num_ingested_files = sum(1 for p in papers if p["ingested"])
    num_chunks = 0

    if num_ingested_files:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        _, num_chunks = ingest_domain(domain, splitter)
        warm_bm25_cache([domain])

    return {
        "domain": domain,
        "papers": papers,
        "num_found": len(papers),
        "num_ingested": num_ingested_files,
        "num_chunks": num_chunks,
    }
