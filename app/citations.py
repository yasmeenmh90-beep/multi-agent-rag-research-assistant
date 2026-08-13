"""
Formats paper metadata (from paper_search.py) into citation strings.
Two styles covers most thesis/university requirements - APA is the most
common in social sciences/general academia, IEEE is standard in CS/engineering.
"""


def _format_authors_apa(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"
    formatted = []
    for name in authors:
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join(f"{p[0]}." for p in parts[:-1])
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(name)

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def format_apa(paper: dict) -> str:
    authors = _format_authors_apa(paper.get("authors", []))
    year = paper.get("year") or "n.d."
    title = paper.get("title", "Untitled").rstrip(".")
    venue = paper.get("venue", "")

    citation = f"{authors} ({year}). {title}."
    if venue:
        citation += f" {venue}."
    if paper.get("doi"):
        citation += f" https://doi.org/{paper['doi']}"
    elif paper.get("arxiv_id"):
        citation += f" arXiv:{paper['arxiv_id']}"
    return citation


def format_ieee(paper: dict, index: int | None = None) -> str:
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown Author"
    else:
        initials_names = []
        for name in authors:
            parts = name.strip().split()
            if len(parts) >= 2:
                initials = ". ".join(p[0] for p in parts[:-1]) + "."
                initials_names.append(f"{initials} {parts[-1]}")
            else:
                initials_names.append(name)
        author_str = ", ".join(initials_names)

    title = paper.get("title", "Untitled").rstrip(".")
    venue = paper.get("venue", "")
    year = paper.get("year") or "n.d."

    prefix = f"[{index}] " if index is not None else ""
    citation = f'{prefix}{author_str}, "{title},"'
    if venue:
        citation += f" {venue},"
    citation += f" {year}."
    if paper.get("doi"):
        citation += f" doi: {paper['doi']}."
    elif paper.get("arxiv_id"):
        citation += f" arXiv:{paper['arxiv_id']}."
    return citation


def format_bibliography(papers: list[dict], style: str = "apa") -> list[str]:
    """Returns a list of formatted citation strings, one per paper, in the
    same order as `papers` (so callers can pair index -> in-text marker)."""
    if style == "ieee":
        return [format_ieee(p, i + 1) for i, p in enumerate(papers)]
    return [format_apa(p) for p in papers]
