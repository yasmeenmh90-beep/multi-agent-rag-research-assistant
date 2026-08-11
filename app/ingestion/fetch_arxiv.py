"""
Downloads PDFs from arXiv for a given search query and saves them into
data/documents/<domain>/, ready for `python -m app.ingestion.ingest`.

Usage:
    python -m app.ingestion.fetch_arxiv --query "retrieval augmented generation" --domain ai_papers --max 120
    python -m app.ingestion.fetch_arxiv --query "multi-agent systems" --domain multi_agent --max 60

Notes:
- Uses arXiv's public Atom API (export.arxiv.org) - no API key needed.
- Sleeps between requests to stay well within arXiv's rate-limit guidance.
- Skips a paper if the PDF fails to download instead of aborting the batch.
"""
import argparse
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import DOCUMENTS_DIR

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
BATCH_SIZE = 50          # arXiv API max per request
REQUEST_DELAY_SEC = 3    # be polite to arXiv's servers


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    return slug.strip("_")[:80]


def fetch_page(query: str, start: int, batch_size: int) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": start,
        "max_results": batch_size,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else "untitled"

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        if pdf_url:
            entries.append({"title": title, "pdf_url": pdf_url})

    return entries


def download_pdf(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  failed to download {url}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk-download arXiv papers for the RAG corpus.")
    parser.add_argument("--query", required=True, help='e.g. "retrieval augmented generation"')
    parser.add_argument("--domain", required=True, help="subfolder under data/documents/ to save into")
    parser.add_argument("--max", type=int, default=100, help="total number of papers to fetch")
    args = parser.parse_args()

    out_dir = DOCUMENTS_DIR / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded, start = 0, 0
    print(f"Searching arXiv for '{args.query}' -> saving into {out_dir}\n")

    while downloaded < args.max:
        remaining = args.max - downloaded
        batch_size = min(BATCH_SIZE, remaining)

        entries = fetch_page(args.query, start, batch_size)
        if not entries:
            print("No more results from arXiv.")
            break

        for entry in entries:
            if downloaded >= args.max:
                break

            filename = f"{slugify(entry['title'])}.pdf"
            dest = out_dir / filename

            if dest.exists():
                print(f"  skip (already exists): {filename}")
                continue

            ok = download_pdf(entry["pdf_url"], dest)
            if ok:
                downloaded += 1
                print(f"  [{downloaded}/{args.max}] saved: {entry['title'][:70]}")

            time.sleep(REQUEST_DELAY_SEC)

        start += batch_size

    print(f"\nDone. {downloaded} PDFs saved to {out_dir}")
    print("Next: python -m app.ingestion.ingest")


if __name__ == "__main__":
    main()
