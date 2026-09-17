"""Article fetching and text extraction.

Fetches web pages and extracts readable text content.
Used to get full article text for LLM analysis.
"""

import logging
import re

from bs4 import BeautifulSoup

from config import MAX_ARTICLE_CHARS, REQUEST_TIMEOUT
from src.net import http_get

logger = logging.getLogger("ai_research_bot")


def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML content.

    Removes scripts, styles, navigation, and other non-content elements.
    Returns clean paragraph text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        element.decompose()

    paragraphs = []

    for tag in soup.find_all(["article", "main", "p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) >= 40:
            paragraphs.append(text)

    text = "\n".join(paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_article(url: str) -> str | None:
    """Fetch a URL and extract its readable text content.

    Returns extracted text (truncated to MAX_ARTICLE_CHARS) or None on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIResearchBot/1.0)"
    }

    try:
        response = http_get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return None

        text = extract_text_from_html(response.text)

        if len(text) < 300:
            return None

        return text[:MAX_ARTICLE_CHARS]

    except Exception as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        return None
