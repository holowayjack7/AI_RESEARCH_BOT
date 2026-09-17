"""Web search module using Tavily API.

Searches the web for AI and agentic engineering news.
Uses curated search queries targeting trusted sources.
"""

import logging

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
        data = http_post_json(url, payload, timeout=20)
        results = data.get("results", [])
        logger.info(f"Tavily: {len(results)} results")
        return results
    except Exception as e:
        # Retries/backoff happen inside src.net; a persistent failure
        # only costs this one query, never the whole run.
        logger.warning(f"Tavily query failed: {e}")
        return []


def collect_search_results(api_key: str) -> list[dict]:
    """Run all search queries and collect results.

    Returns a list of raw result dicts with title, url, content, score, domain.
    """
    logger.info("Searching Tavily...")

    all_results = []

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

            domain = domain_from_url(url)

            all_results.append({
                "title": result.get("title", "").strip(),
                "url": url,
                "content": result.get("content", "").strip(),
                "score": result.get("score", 0),
                "domain": domain,
                "source": "web",
                "published": "",
                "dedup_key": "",
            })

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
