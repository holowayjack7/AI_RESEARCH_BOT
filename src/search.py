"""Web search module using Tavily API.

Searches the web for AI and agentic engineering news.
Uses curated search queries targeting trusted sources.
"""

import logging
import re
from urllib.parse import urlparse

from src.net import http_post_json

logger = logging.getLogger("ai_research_bot")


# ============================================================
# SEARCH QUERIES — focused on AI agents, LLMs, tools, opportunities
# ============================================================

SEARCH_QUERIES = [
    (
        "AI agents agent engineering frameworks MCP tool calling "
        "agent infrastructure latest releases developers"
    ),
    (
        "latest LLM APIs AI developer tools coding agents "
        "AI IDEs open source models developer releases"
    ),
    (
        "RAG retrieval agent memory context engineering "
        "agent evaluation benchmarks tool use MCP latest"
    ),
    (
        "major AI model releases AI research open source models "
        "AI engineering infrastructure latest"
    ),
    (
        "AI coding agents autonomous coding software engineering "
        "Claude Code Cursor Codex agent workflows latest"
    ),
    (
        "AI opportunities students teenagers 16+ competitions "
        "hackathons grants fellowships programs developers"
    ),
    (
        "AI startups open source developer infrastructure "
        "new AI tools APIs SDKs repositories latest"
    ),
    (
        "important AI research papers agents reasoning inference "
        "training evaluation multimodal models latest"
    ),
]


# ============================================================
# TRUSTED DOMAINS — only accept results from these sources
# ============================================================

TRUSTED_DOMAINS = {
    "openai.com",
    "anthropic.com",
    "claude.com",
    "blog.google",
    "deepmind.google",
    "ai.google.dev",
    "developers.googleblog.com",
    "research.google",
    "microsoft.com",
    "azure.microsoft.com",
    "aws.amazon.com",
    "x.ai",
    "docs.x.ai",
    "mistral.ai",
    "deepseek.com",
    "nvidia.com",
    "meta.com",
    "ai.meta.com",
    "github.com",
    "huggingface.co",
    "arxiv.org",
    "kaggle.com",
    "mlh.io",
    "hackclub.com",
    "developers.google.com",
    "cloud.google.com",
    "together.ai",
    "cohere.com",
    "replicate.com",
    "ollama.com",
    "langchain.com",
    "llamaindex.ai",
    "runloop.ai",
    "agentprotocol.ai",
    "openagents.com",
    "aits.docs.buildwithfern.com",
}


# ============================================================
# EXCLUDED DOMAINS — low-value or spammy sources
# ============================================================

EXCLUDED_DOMAINS = {
    "ainotdie.com",
    "claudelog.com",
    "lumichats.com",
    "stocktwits.com",
    "labmanager.com",
}


# ============================================================
# LOW-VALUE FILTER — drop clickbait, fluff, and noise
# ============================================================

# Compiled once; matched case-insensitively against title + content
LOW_VALUE_PATTERNS = [
    re.compile(
        r"you won'?t believe|shocking|insane|mind-?blowing|"
        r"game-?changing|revolutionary|jaw-?dropping|unbelievable",
        re.IGNORECASE,
    ),
    re.compile(r"\btop \d+\b|\bbest \d+\b|\d+ things|\d+ ways|\d+ reasons", re.IGNORECASE),
    re.compile(
        r"coupon|promo code|discount code|giveaway|free money|lottery",
        re.IGNORECASE,
    ),
    # Financial noise the reader explicitly does not want
    re.compile(
        r"stock (?:soars|plunges|jumps|dumps|rallies|slides)|"
        r"share price|earnings call|quarterly revenue|market cap",
        re.IGNORECASE,
    ),
    # Unverified noise
    re.compile(r"\brumou?r\b|\bleaked\b|speculation|allegedly", re.IGNORECASE),
]

# Path fragments that mark official/technical material worth boosting
OFFICIAL_RELEASE_HINTS = re.compile(
    r"releases?|release-?notes|changelog|changelogs|/docs|/blog|"
    r"announc|launch|introduc|now-available|generally-available",
    re.IGNORECASE,
)

MIN_WEB_SUMMARY_CHARS = 80

# Results must carry concrete technical data or code: at least one
# digit or one technical term. Papers, release notes, docs, and
# changelogs always do; marketing fluff and vague summaries don't.
TECH_SIGNAL_PATTERN = re.compile(
    r"\d|"
    r"\b(?:api|sdk|cli|benchmark|token|latency|throughput|context "
    r"window|parameters?|fine-?tun|inference|open[- ]?source|repo|"
    r"repositor|release|changelog|migrat|integrat|deploy|framework|"
    r"model|agent|rag|llm|prompt|code|python|typescript|rust|golang|"
    r"github|npm|pip|docker|kubernetes|gpu|weights|checkpoint)\b",
    re.IGNORECASE,
)


def is_low_value_result(title: str, content: str) -> tuple[bool, str]:
    """Judge whether a web result is clickbait/fluff worth dropping.

    Returns (is_low_value, reason). Papers and official docs rarely
    trip these patterns; noisy aggregator copy does.
    """
    title = (title or "").strip()
    content = (content or "").strip()

    if not title or len(title) < 15:
        return True, "empty/thin title"

    if len(content) < MIN_WEB_SUMMARY_CHARS:
        return True, f"summary too short ({len(content)} chars)"

    text = f"{title}. {content}"
    for pattern in LOW_VALUE_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"low-value pattern: '{match.group(0)}'"

    # Hype punctuation ("?!?", "!!!") or emoji-stuffed titles
    if re.search(r"[?!]{2,}", title) or len(re.findall(r"[\U0001F300-\U0001FAFF]", title)) >= 2:
        return True, "hype punctuation/emoji title"

    # Shouty titles: 3+ long ALL-CAPS words (excludes normal acronyms
    # like API/LLM which are <= 4 chars)
    long_caps = [w for w in re.findall(r"[A-Z]{5,}", title)]
    if len(long_caps) >= 3:
        return True, "shouty ALL-CAPS title"

    # No concrete technical data or code -> drop before it can pollute
    # the analysis stage
    if not TECH_SIGNAL_PATTERN.search(f"{title}. {content}"):
        return True, "no concrete technical data or code"

    return False, ""


def _boost_official_release(url: str, score: float) -> float:
    """Small score bump for official release/docs/changelog URLs."""
    path = urlparse(url).path.lower()
    if OFFICIAL_RELEASE_HINTS.search(path):
        return min(1.0, score + 0.05)
    return score


def domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    from urllib.parse import urlparse

    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        return domain
    except Exception:
        return ""


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    from urllib.parse import urlparse

    if not url:
        return ""

    url = url.strip()
    parsed = urlparse(url)

    if not parsed.scheme:
        return ""

    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return clean.rstrip("/")


def is_trusted_domain(url: str) -> bool:
    """Check if URL is from a trusted source."""
    domain = domain_from_url(url)

    if not domain:
        return False

    if domain in EXCLUDED_DOMAINS:
        return False

    return any(
        domain == trusted or domain.endswith("." + trusted)
        for trusted in TRUSTED_DOMAINS
    )


def search_tavily(query: str, api_key: str) -> list[dict]:
    """Execute a single Tavily search query.

    Returns raw search results from the API.
    """
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "topic": "news",
        "time_range": "week",
        "max_results": 8,
        "include_domains": list(TRUSTED_DOMAINS),
        "exclude_domains": list(EXCLUDED_DOMAINS),
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        from config import HTTP_MAX_ATTEMPTS

        data = http_post_json(url, payload, timeout=20, max_attempts=HTTP_MAX_ATTEMPTS)
        results = data.get("results", [])
        logger.info(f"Tavily: {len(results)} results")
        return results
    except Exception as e:
        # Retries with exponential backoff happen inside src.net (see
        # HTTP_MAX_ATTEMPTS); a persistent failure only costs this one
        # query, never the whole run.
        logger.warning(f"Tavily query failed: {e}")
        return []


def collect_search_results(api_key: str) -> list[dict]:
    """Run all search queries and collect results.

    Returns a list of raw result dicts with title, url, content, score, domain.
    Low-value/clickbait results are dropped before they can pollute
    the analysis stage.
    """
    logger.info("Searching Tavily...")

    all_results = []
    dropped = 0

    for index, query in enumerate(SEARCH_QUERIES, start=1):
        logger.info(f"Search {index}/{len(SEARCH_QUERIES)}: {query[:60]}...")

        results = search_tavily(query, api_key)
        logger.info(f"  Found {len(results)} results")

        for result in results:
            url = normalize_url(result.get("url", ""))

            if not url:
                continue

            if not is_trusted_domain(url):
                continue

            title = result.get("title", "").strip()
            content = result.get("content", "").strip()

            low_value, reason = is_low_value_result(title, content)
            if low_value:
                dropped += 1
                logger.debug(f"Dropped low-value result ({reason}): {title[:60]}")
                continue

            domain = domain_from_url(url)

            all_results.append({
                "title": title,
                "url": url,
                "content": content,
                "score": _boost_official_release(url, result.get("score", 0)),
                "domain": domain,
                "source": "web",
                "published": "",
                "dedup_key": "",
            })

    if dropped:
        logger.info(f"Dropped {dropped} low-value/clickbait results")

    logger.info(f"Total raw results: {len(all_results)}")
    return all_results


def deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove duplicate results by URL, keeping highest scored."""
    unique = {}

    for result in results:
        url = result["url"]
        if url not in unique:
            unique[url] = result
        elif result["score"] > unique[url]["score"]:
            unique[url] = result

    deduped = list(unique.values())
    logger.info(f"After URL dedup: {len(deduped)}")
    return deduped
