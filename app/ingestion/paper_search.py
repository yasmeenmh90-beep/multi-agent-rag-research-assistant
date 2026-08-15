"""
Searches arXiv and Semantic Scholar for papers on a topic and returns a
normalized list of metadata dicts - the input to fetch_and_ingest_topic.py.

Two sources because they cover different ground:
- arXiv: CS/physics/math preprints, always has a direct PDF link, no API key.
- Semantic Scholar: much broader field coverage (medicine, social science,
  etc.) and includes citation counts, but a PDF link is only present when
  the paper is actually open-access (openAccessPdf field) - many results
  will have metadata but no downloadable PDF, which is expected.

Both APIs are public and don't require a key for this volume of usage.
"""
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,year,authors,venue,externalIds,openAccessPdf,citationCount"
# Optional - unauthenticated requests share a rate limit across every
# anonymous caller on the same IP, which on a platform like Render means
# sharing it with every other customer's traffic too, not just this app's.
# An API key (free: https://www.semanticscholar.org/product/api#api-key)
# gets its own dedicated quota instead. Works fine without one, just more
# likely to hit 429s.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _build_arxiv_query(topic: str) -> str:
    """A plain multi-word 'all:topic words here' query is interpreted as an
    OR of the individual words by arXiv's search - a paper matching just one
    common word (e.g. "system") can rank alongside genuine topic matches.
    ANDing every word together keeps results actually on-topic."""
    words = [w for w in topic.split() if w]
    return " AND ".join(f"all:{w}" for w in words) or f"all:{topic}"


def search_arxiv(topic: str, max_results: int = 15) -> list[dict]:
    params = {
        "search_query": _build_arxiv_query(topic),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",   # topic match matters more than recency here -
        "sortOrder": "descending",   # sorting by date alone returned unrelated papers
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  arXiv search failed: {exc}")
        return []

    root = ET.fromstring(raw)
    results = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled"

        summary_el = entry.find("atom:summary", ATOM_NS)
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""

        published_el = entry.find("atom:published", ATOM_NS)
        year = published_el.text[:4] if published_el is not None else ""

        authors = [
            a.find("atom:name", ATOM_NS).text
            for a in entry.findall("atom:author", ATOM_NS)
            if a.find("atom:name", ATOM_NS) is not None
        ]

        pdf_url = None
        arxiv_id = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
            if link.attrib.get("rel") == "alternate":
                arxiv_id = link.attrib.get("href", "").rsplit("/", 1)[-1]

        results.append({
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "venue": "arXiv",
            "pdf_url": pdf_url,
            "source_api": "arxiv",
            "arxiv_id": arxiv_id,
            "doi": None,
        })

    return results


def search_semantic_scholar(topic: str, max_results: int = 15) -> list[dict]:
    params = {
        "query": topic,
        "limit": max_results,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    url = f"{SEMANTIC_SCHOLAR_API}?{urllib.parse.urlencode(params)}"

    headers = {"User-Agent": "Mozilla/5.0"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    data = None
    for attempt in range(2):  # one retry specifically for 429s, which are
        # often just a momentary shared-IP burst rather than a hard block
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                import json
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                print("  Semantic Scholar rate-limited, waiting 5s before one retry ...")
                time.sleep(5)
                continue
            print(f"  Semantic Scholar search failed: {exc}")
            return []
        except Exception as exc:  # noqa: BLE001
            print(f"  Semantic Scholar search failed: {exc}")
            return []

    if data is None:
        return []

    results = []
    for paper in data.get("data", []):
        open_access = paper.get("openAccessPdf") or {}
        results.append({
            "title": paper.get("title") or "Untitled",
            "authors": [a.get("name", "") for a in paper.get("authors", [])],
            "year": str(paper.get("year") or ""),
            "abstract": paper.get("abstract") or "",
            "venue": paper.get("venue") or "",
            "pdf_url": open_access.get("url"),  # None if not open-access
            "source_api": "semantic_scholar",
            "arxiv_id": None,
            "doi": (paper.get("externalIds") or {}).get("DOI"),
            "citation_count": paper.get("citationCount"),
        })

    return results


def search_topic(topic: str, max_per_source: int = 15) -> list[dict]:
    """Searches both sources and dedupes by normalized title, preferring
    the arXiv entry when the same paper appears in both (it always has a
    direct PDF link, Semantic Scholar's often doesn't)."""
    arxiv_results = search_arxiv(topic, max_per_source)
    time.sleep(1)  # be polite between the two API calls
    ss_results = search_semantic_scholar(topic, max_per_source)

    seen_titles = set()
    combined = []

    for paper in arxiv_results + ss_results:
        key = _normalize_title(paper["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        combined.append(paper)

    return combined
