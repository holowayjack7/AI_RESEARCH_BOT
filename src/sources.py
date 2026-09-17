"""Additional research sources: arXiv and Hugging Face Daily Papers.

Both sources require no API keys. Results are returned in the same
candidate-dict shape used by the Tavily search module so the rest of
the pipeline can treat them uniformly:

    {title, url, content, score, domain, source, published, dedup_key}
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from src.net import http_get

logger = logging.getLogger("ai_research_bot")

# arXiv categories most relevant to the reader's profile
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.MA", "cs.IR"]

ARXIV_API_URL = "https://export.arxiv.org/api/query"
HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"

ARXIV_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def within_days(date_str: str, days: int) -> bool:
    """Check an ISO date string is within the last N days.

    Unparseable/missing dates are kept (returns True) — better to
    let the LLM stage judge relevance than to silently drop items.
    """
    if not date_str:
        return True
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - dt
        return age.total_seconds() <= days * 86400
    except Exception:
        return True


def _candidate_from_paper(
    *,
    source: str,
    title: str,
    abstract: str,
    url: str,
    arxiv_id: str,
    published: str,
    authors: list[str],
    score: float,
    extra: str = "",
) -> dict:
    """Build a uniform candidate dict from a paper record."""
    content = abstract
    if authors:
        content = f"Authors: {', '.join(authors[:6])}\n\n{abstract}"
    if extra:
        content = f"{extra}\n\n{content}"

    return {
        "title": title.strip(),
        "url": url,
        "content": content.strip(),
        "score": score,
        "domain": "arxiv.org" if source == "arxiv" else "huggingface.co",
        "source": source,
        "published": published,
        "dedup_key": f"arxiv:{arxiv_id}",
    }


def fetch_arxiv_papers(max_results: int = 15, max_age_days: int = 3) -> list[dict]:
    """Fetch the latest papers from the arXiv API (Atom XML)."""
    query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = http_get(ARXIV_API_URL, params=params, timeout=30)
        root = ET.fromstring(response.text)
    except Exception as e:
        logger.warning(f"arXiv fetch failed: {e}")
        return []

    papers = []

    for entry in root.findall(f"{ARXIV_ATOM_NS}entry"):
        raw_id = entry.findtext(f"{ARXIV_ATOM_NS}id", "")  # e.g. http://arxiv.org/abs/2609.09778v1
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            continue

        base_id = re.sub(r"v\d+$", "", arxiv_id)
        title = " ".join(entry.findtext(f"{ARXIV_ATOM_NS}title", "").split())
        abstract = " ".join(entry.findtext(f"{ARXIV_ATOM_NS}summary", "").split())
        published = entry.findtext(f"{ARXIV_ATOM_NS}published", "")
        authors = [
            a.findtext(f"{ARXIV_ATOM_NS}name", "")
            for a in entry.findall(f"{ARXIV_ATOM_NS}author")
        ]
        categories = [
            c.get("term", "") for c in entry.findall(f"{ARXIV_ATOM_NS}category")
        ]

        if not within_days(published, max_age_days):
            continue

        papers.append(_candidate_from_paper(
            source="arxiv",
            title=title,
            abstract=abstract,
            url=f"https://arxiv.org/abs/{base_id}",
            arxiv_id=base_id,
            published=published,
            authors=authors,
            score=0.9,
            extra=f"Categories: {', '.join(c for c in categories if c)}",
        ))

    logger.info(f"arXiv: {len(papers)} fresh papers (last {max_age_days} days)")
    return papers


def fetch_hf_papers(max_results: int = 15, max_age_days: int = 3) -> list[dict]:
    """Fetch trending papers from Hugging Face Daily Papers (JSON)."""
    try:
        response = http_get(HF_DAILY_PAPERS_URL, timeout=30)
        items = response.json()
    except Exception as e:
        logger.warning(f"Hugging Face papers fetch failed: {e}")
        return []

    if not isinstance(items, list):
        logger.warning("Hugging Face papers: unexpected response shape")
        return []

    papers = []

    for item in items[: max_results * 2]:
        paper = item.get("paper") or {}
        raw_id = str(paper.get("id", "") or item.get("id", "")).strip()
        title = str(paper.get("title", "") or "").strip()

        if not raw_id or not title:
            continue

        # HF paper ids are arXiv ids -> cross-source dedup works
        base_id = re.sub(r"v\d+$", "", raw_id)
        abstract = str(paper.get("summary", "") or "").replace("\n", " ").strip()
        published = str(
            paper.get("publishedAt")
            or item.get("publishedAt")
            or paper.get("upvoted_at")
            or ""
        )
        authors = [
            a.get("name", "")
            for a in (paper.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        upvotes = paper.get("upvotes") or item.get("upvotes") or 0

        if not within_days(published, max_age_days):
            continue

        try:
            score = min(1.0, 0.5 + float(upvotes) / 100.0)
        except (TypeError, ValueError):
            score = 0.6

        papers.append(_candidate_from_paper(
            source="hf_papers",
            title=title,
            abstract=abstract,
            url=f"https://huggingface.co/papers/{base_id}",
            arxiv_id=base_id,
            published=published,
            authors=authors,
            score=round(score, 2),
            extra=f"Upvotes: {upvotes}",
        ))

        if len(papers) >= max_results:
            break

    logger.info(f"Hugging Face: {len(papers)} trending papers (last {max_age_days} days)")
    return papers


def collect_all_sources(tavily_key: str) -> list[dict]:
    """Collect candidates from all enabled sources.

    A failure in any single source is logged and skipped — the run
    continues with whatever the other sources returned.
    """
    from config import (
        ENABLE_ARXIV, ENABLE_HF_PAPERS,
        ARXIV_MAX_RESULTS, HF_MAX_RESULTS, PAPER_MAX_AGE_DAYS,
    )

    all_results: list[dict] = []

    # --- Tavily web search (news + trusted domains) ---
    if tavily_key:
        from src.search import collect_search_results
        all_results.extend(collect_search_results(tavily_key))
    else:
        logger.info("Tavily key not set — skipping web search")

    # --- arXiv papers ---
    if ENABLE_ARXIV:
        all_results.extend(
            fetch_arxiv_papers(ARXIV_MAX_RESULTS, PAPER_MAX_AGE_DAYS)
        )

    # --- Hugging Face Daily Papers ---
    if ENABLE_HF_PAPERS:
        all_results.extend(
            fetch_hf_papers(HF_MAX_RESULTS, PAPER_MAX_AGE_DAYS)
        )

    logger.info(f"All sources collected: {len(all_results)} raw results")
    return all_results
