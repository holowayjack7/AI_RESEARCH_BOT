"""Research pipeline — full implementation.

Orchestrates: search → collect → deduplicate → filter → fetch →
analyze → validate → publish → deliver

Sources: Tavily web search + arXiv + Hugging Face Daily Papers
(configurable via feature flags). The pipeline is resilient: any
single source failing never aborts the run.
"""

import hashlib
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone

from config import (
    MAX_CANDIDATES_FOR_RESEARCH,
    MIN_SOURCE_QUALITY,
    MIN_IMPORTANCE,
    MIN_RELEVANCE,
    MIN_ACTIONABILITY,
    MIN_CONFIDENCE,
)
from src.search import deduplicate_results, domain_from_url
from src.sources import collect_all_sources
from src.fetcher import fetch_article
from src.analyze import analyze_with_gemini, parse_report, ResearchReport
from src.deliver import (
    build_no_news_message,
    build_telegram_message,
    send_telegram,
)
from src.publish import publish_report

logger = logging.getLogger("ai_research_bot")


# ============================================================
# SOURCE QUALITY SCORING
# ============================================================

SOURCE_QUALITY_MAP = {
    "openai.com": 10, "anthropic.com": 10, "claude.com": 10,
    "ai.google.dev": 10, "deepmind.google": 10, "research.google": 10,
    "developers.googleblog.com": 10,
    "microsoft.com": 10, "azure.microsoft.com": 10,
    "aws.amazon.com": 10, "x.ai": 10, "mistral.ai": 10,
    "deepseek.com": 10, "nvidia.com": 10, "meta.com": 10, "ai.meta.com": 10,
    "github.com": 8, "huggingface.co": 8, "arxiv.org": 8,
    "kaggle.com": 8, "mlh.io": 8, "hackclub.com": 8,
    "developers.google.com": 9, "cloud.google.com": 9,
    "together.ai": 8, "cohere.com": 8, "replicate.com": 8,
    "ollama.com": 8, "langchain.com": 8, "llamaindex.ai": 8,
}


def source_quality(domain: str) -> int:
    """Score a domain's source quality (1-10)."""
    return SOURCE_QUALITY_MAP.get(domain, 5)


# ============================================================
# DEDUP KEYS
# ============================================================

def dedup_key_from_url(url: str) -> str:
    """Derive a cross-source dedup key from a URL (arXiv id based).

    The same paper can surface via arxiv.org/abs, /html, /pdf or
    huggingface.co/papers — all must map to one key.
    """
    match = re.search(
        r"(?:arxiv\.org/(?:abs|html|pdf)/|huggingface\.co/papers/)"
        r"([0-9]{4}\.[0-9]{4,5})",
        url or "",
    )
    return f"arxiv:{match.group(1)}" if match else ""


def collapse_by_dedup_key(results: list[dict]) -> list[dict]:
    """Collapse overlapping stories sharing a dedup key (e.g. the same
    arXiv paper surfaced by arXiv API, Hugging Face, and Tavily).
    Keeps the highest-scored variant of each story; the kept variant
    records its siblings' URLs under "collapsed_with" so validation
    can still accept them as supporting links of the same story.
    """
    groups: dict[str, list[dict]] = {}
    no_key: list[dict] = []

    for result in results:
        key = result.get("dedup_key") or dedup_key_from_url(result["url"])
        if not key:
            no_key.append(result)
            continue
        groups.setdefault(key, []).append(result)

    collapsed = list(no_key)
    for key, members in groups.items():
        best = max(members, key=lambda r: r.get("score", 0))
        siblings = [m["url"] for m in members if m is not best]
        if siblings:
            best["collapsed_with"] = list(
                dict.fromkeys(best.get("collapsed_with", []) + siblings)
            )
        collapsed.append(best)

    if len(collapsed) < len(results):
        logger.info(f"Dedup keys collapsed {len(results)} -> {len(collapsed)} results")
    return collapsed


def _collapsed_extra_urls(candidates: list[dict]) -> set[str]:
    """URLs of collapsed same-story siblings across all candidates.

    Only true sibling URLs are whitelisted — never URLs filtered out
    for other reasons (already seen, low quality).
    """
    extra: set[str] = set()
    for candidate in candidates:
        extra.update(candidate.get("collapsed_with") or [])
    return extra


# ============================================================
# CANDIDATE SELECTION
# ============================================================

def prepare_candidates(results: list[dict], state: dict) -> list[dict]:
    """Filter and rank candidates for deep analysis.

    Skips anything already seen by URL or by dedup key (e.g. the
    same arXiv paper seen earlier through a different URL).
    """
    seen_urls = set(state.get("seen_urls", []))
    seen_keys = set(state.get("seen_dedup_keys", []))

    candidates = []

    for result in results:
        url = result["url"]

        if url in seen_urls:
            continue

        key = result.get("dedup_key") or dedup_key_from_url(url)
        if key and key in seen_keys:
            logger.info(f"Skipped (dedup key already seen): {key}")
            continue

        domain = result.get("domain", "") or domain_from_url(url)
        quality = source_quality(domain)

        if quality < MIN_SOURCE_QUALITY:
            continue

        result["source_quality"] = quality
        if key:
            result["dedup_key"] = key
        candidates.append(result)

    # Sort by quality then score
    candidates.sort(key=lambda x: (x["source_quality"], x.get("score", 0)), reverse=True)
    candidates = candidates[:MAX_CANDIDATES_FOR_RESEARCH]

    logger.info(f"Research candidates: {len(candidates)}")
    for i, c in enumerate(candidates, 1):
        logger.info(f"  {i}. {c['title'][:80]} [{c.get('source', c['domain'])}]")

    return candidates


# ============================================================
# REPORT VALIDATION
# ============================================================

def make_event_id(title: str, urls: list[str]) -> str:
    """Build a stable event id from the title and its URLs."""
    digest = hashlib.sha256(
        (title + "|" + "|".join(urls)).encode("utf-8")
    ).hexdigest()
    return f"evt-{digest[:12]}"


def validate_report(
    report: ResearchReport,
    candidates: list[dict],
    state: dict | None = None,
    extra_allowed_urls: set[str] | None = None,
) -> ResearchReport:
    """Validate and filter Gemini's output against actual candidates.

    Skips events already reported in previous runs (seen_event_ids,
    plus arXiv-id dedup keys) and backfills missing event ids with a
    stable hash.

    extra_allowed_urls lets callers whitelist URLs that were part of
    this run but not kept as candidates (e.g. a cross-source duplicate
    that got collapsed into a kept candidate — its URL still belongs
    to the same story).
    """
    state = state or {}
    allowed_urls = {c["url"] for c in candidates}
    if extra_allowed_urls:
        allowed_urls |= extra_allowed_urls
    seen_event_ids = set(state.get("seen_event_ids", []))
    seen_keys = set(state.get("seen_dedup_keys", []))

    valid_events = []

    for event in report.events:
        # Must have a valid primary URL
        if event.primary_url not in allowed_urls:
            logger.warning(f"Rejected: invalid URL -> {event.primary_url}")
            continue

        # Filter supporting URLs
        event.supporting_urls = [u for u in event.supporting_urls if u in allowed_urls]

        # Stable event id (backfill if Gemini left it empty)
        event.event_id = (
            event.event_id.strip()
            or make_event_id(
                event.title,
                [event.primary_url] + event.supporting_urls,
            )
        )

        # Already reported in a previous run
        if event.event_id in seen_event_ids:
            logger.info(f"Skipped (already seen): {event.event_id}")
            continue

        # Same underlying paper reported before (any URL variant)
        event_key = dedup_key_from_url(event.primary_url)
        if event_key and event_key in seen_keys:
            logger.info(f"Skipped (dedup key already seen): {event_key}")
            continue

        # Apply quality thresholds
        if event.importance < MIN_IMPORTANCE:
            continue
        if event.relevance < MIN_RELEVANCE:
            continue
        if event.actionability < MIN_ACTIONABILITY:
            continue
        if event.confidence < MIN_CONFIDENCE:
            continue

        # Cap source quality at domain reality
        domain = domain_from_url(event.primary_url)
        event.source_quality = min(event.source_quality, source_quality(domain))

        if event.source_quality < MIN_SOURCE_QUALITY:
            continue

        valid_events.append(event)

    # Sort by importance, relevance, actionability, confidence
    valid_events.sort(
        key=lambda e: (e.importance, e.relevance, e.actionability, e.confidence),
        reverse=True,
    )

    report.events = valid_events[:15]
    return report


# ============================================================
# PERSISTENT STATE UPDATE
# ============================================================

def update_state(state: dict, report: ResearchReport, candidates: list[dict]) -> dict:
    """Merge this run's results into persistent state.

    Caps: seen_urls (2000), seen_event_ids (1000), seen_dedup_keys
    (1000), events (100), reports (14), learning_queue (20).

    seen_urls records ONLY the URLs that were actually reported
    (primary + supporting). Analyzed-but-unreported candidates are
    deliberately NOT marked seen, so they can resurface next run while
    still fresh (paper feeds cover ~3 days, news ~1 week).
    """
    state.setdefault("seen_urls", [])
    state.setdefault("seen_event_ids", [])
    state.setdefault("seen_dedup_keys", [])
    state.setdefault("events", [])
    state.setdefault("topics", {})
    state.setdefault("reports", [])
    state.setdefault("learning_queue", [])

    delivered_urls = []
    for event in report.events:
        delivered_urls.append(event.primary_url)
        delivered_urls.extend(event.supporting_urls)

    state["seen_urls"] = list(
        dict.fromkeys(
            state["seen_urls"] + [u for u in delivered_urls if u]
        )
    )[-2000:]

    for event in report.events:
        state["seen_event_ids"].append(event.event_id)
        state["events"].append(asdict(event))

        topic = event.category
        state["topics"][topic] = state["topics"].get(topic, 0) + 1

        key = dedup_key_from_url(event.primary_url)
        if key:
            state["seen_dedup_keys"].append(key)

    state["seen_event_ids"] = list(
        dict.fromkeys(state["seen_event_ids"])
    )[-1000:]

    state["seen_dedup_keys"] = list(
        dict.fromkeys(state["seen_dedup_keys"])
    )[-1000:]

    state["events"] = state["events"][-100:]

    state["reports"].append({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "summary": report.executive_summary,
        "events": [asdict(e) for e in report.events],
        "trends": report.trends,
    })
    state["reports"] = state["reports"][-14:]

    state["learning_queue"] = report.learn_next[-20:]

    return state


# ============================================================
# MAIN PIPELINE
# ============================================================

def _send_no_news_heartbeat(
    state: dict,
    telegram_token: str,
    telegram_chat_id: str,
    candidates_reviewed: int,
) -> bool:
    """Send the daily 'no new intelligence' heartbeat.

    The user expects a report every day — quiet days still get a
    short proof-of-life message. Never raises: a heartbeat failure
    is logged, not fatal.
    """
    if not (telegram_token and telegram_chat_id):
        logger.info("Telegram not configured — no-news heartbeat skipped")
        return False

    try:
        message = build_no_news_message(state, candidates_reviewed)
        sent = send_telegram(message, telegram_token, telegram_chat_id)
        if sent:
            logger.info("✅ No-news heartbeat delivered")
        return sent
    except Exception as e:
        logger.error(f"No-news heartbeat failed: {e}")
        return False

def run_pipeline(
    tavily_key: str,
    gemini_key: str,
    telegram_token: str = "",
    telegram_chat_id: str = "",
    state: dict | None = None,
    gemini_model: str = "gemini-3.6-flash",
) -> bool:
    """Execute the full research pipeline.

    Sources are collected from all enabled providers; report files
    are always exported to disk. Telegram delivery happens only when
    credentials are provided; persistent state is updated ONLY after
    a successful delivery (Telegram send or export-only publish).

    Returns True if the report was delivered/published.
    """
    state = state if state is not None else {}

    # --- Step 1/7: SEARCH + COLLECT (all sources) ---
    logger.info("Step 1/7: Collecting from all sources (web, arXiv, HF)...")
    raw_results = collect_all_sources(tavily_key)

    # --- Step 2/7: DEDUPLICATE (by URL, then by story key) ---
    logger.info("Step 2/7: Deduplicating...")
    unique_results = collapse_by_dedup_key(deduplicate_results(raw_results))

    # --- Step 3/7: FILTER (seen URLs, dedup keys, source quality) ---
    logger.info("Step 3/7: Filtering candidates...")
    candidates = prepare_candidates(unique_results, state)

    if not candidates:
        logger.info("No new research candidates found")
        return _send_no_news_heartbeat(
            state, telegram_token, telegram_chat_id, candidates_reviewed=0
        )

    # --- Step 4/7: FETCH ARTICLES (best effort per candidate) ---
    logger.info("Step 4/7: Fetching article content...")
    fetched = 0
    for candidate in candidates:
        try:
            article_text = fetch_article(candidate["url"])
        except Exception as e:
            logger.warning(f"Fetcher crashed for {candidate['url']}: {e}")
            article_text = None
        if article_text:
            candidate["content"] = article_text
            fetched += 1
    logger.info(f"  Full text fetched for {fetched}/{len(candidates)} candidates")

    # --- Step 5/7: ANALYZE WITH GEMINI ---
    logger.info("Step 5/7: Analyzing with Gemini...")
    report = analyze_with_gemini(candidates, gemini_key, state, gemini_model)

    # --- VALIDATE ---
    # Collapsed cross-source duplicates are still valid URLs for the
    # same story: pass them through so Gemini's supporting_urls survive
    # validation even though they were removed from the candidate list.
    report = validate_report(
        report, candidates, state,
        extra_allowed_urls=_collapsed_extra_urls(candidates),
    )
    logger.info(f"Validated events: {len(report.events)}")

    if not report.events:
        logger.info("No sufficiently important events found")
        return _send_no_news_heartbeat(
            state, telegram_token, telegram_chat_id,
            candidates_reviewed=len(candidates),
        )

    # --- Step 6/7: PUBLISH (Markdown / JSON / newsletter / social) ---
    logger.info("Step 6/7: Publishing report files...")
    try:
        outputs = publish_report(report)
    except Exception as e:
        logger.error(f"Publishing failed: {e}")
        return False

    # --- Step 7/7: DELIVER ---
    telegram_configured = bool(telegram_token and telegram_chat_id)

    if telegram_configured:
        logger.info("Step 7/7: Delivering to Telegram...")
        message = build_telegram_message(report)
        success = send_telegram(message, telegram_token, telegram_chat_id)
    else:
        logger.info("Step 7/7: Telegram not configured — export-only run")
        success = True

    if success:
        # ONLY after successful delivery: update persistent memory
        update_state(state, report, candidates)
        state["last_report_files"] = outputs
        logger.info("✅ Delivery complete — state updated")

    return success


# ============================================================
# SIMULATION MODE (offline end-to-end test)
# ============================================================

SIMULATED_SOURCES = [
    {
        "title": "ROAM++: Self-Organizing Agent Memory via Learned Relations",
        "url": "https://arxiv.org/abs/2609.12001",
        "content": (
            "Authors: A. Researcher, B. Smith\n\n"
            "We introduce ROAM++, an extension of atomic agent memory that "
            "learns semantic relations between memories and reorganizes them "
            "autonomously. On LoCoMo, ROAM++ improves answer accuracy by "
            "34.2 percentage points over compaction baselines while reducing "
            "retrieval latency 22%. We release code and evaluation harnesses."
        ),
        "score": 0.95,
        "domain": "arxiv.org",
        "source": "arxiv",
        "published": "2026-09-15",
        "dedup_key": "arxiv:2609.12001",
    },
    {
        "title": "Same paper via Hugging Face (cross-source duplicate)",
        "url": "https://huggingface.co/papers/2609.12001",
        "content": "Trending paper: ROAM++ agent memory.",
        "score": 0.8,
        "domain": "huggingface.co",
        "source": "hf_papers",
        "published": "2026-09-15",
        "dedup_key": "arxiv:2609.12001",
    },
    {
        "title": "Fixture: open-source coding-agent SDK released",
        "url": "https://github.com/example-labs/fixture-agent-sdk",
        "content": (
            "Fixture Agent SDK is a new open-source Python SDK for building "
            "coding agents: tool calling, MCP client support, sandboxed "
            "execution and evaluation harnesses included. Apache-2.0."
        ),
        "score": 0.9,
        "domain": "github.com",
        "source": "web",
        "published": "2026-09-15",
    },
]

SIMULATED_REPORT_JSON = """
{
  "report_title": "SIMULATION — AI Intelligence Report",
  "executive_summary": "Simulated run validating the full offline pipeline: curation, validation, export, and state tracking.",
  "events": [
    {
      "event_id": "evt-sim-roam-plus",
      "title": "ROAM++: Self-Organizing Agent Memory via Learned Relations",
      "category": "AI Research / Agent Memory",
      "primary_url": "https://arxiv.org/abs/2609.12001",
      "supporting_urls": ["https://huggingface.co/papers/2609.12001"],
      "importance": 9,
      "relevance": 10,
      "actionability": 8,
      "source_quality": 8,
      "confidence": 90,
      "tldr": "A new agent-memory architecture that learns relations between memories, cutting retrieval cost while lifting accuracy sharply.",
      "what_happened": "ROAM++ was published with code and benchmarks.",
      "what_changed": "Memory organization moved from fixed policies to learned semantic relations.",
      "why_it_matters": "Directly applicable to the reader's agent memory roadmap.",
      "key_takeaways": [
        "Learned relations beat fixed heuristics for memory organization",
        "34.2 point accuracy gain over compaction baselines"
      ],
      "technical_architecture": [
        "Atomic memories with learned relation graph",
        "Deterministic reorganization executed after relation inference"
      ],
      "technical_details": ["LoCoMo +34.2pp", "22% lower retrieval latency"],
      "potential_impact": "Changes how long-lived agent memory systems should be designed.",
      "action_type": "LEARN",
      "action": "Read the paper and sketch a minimal relation-classifier over toy memories."
    },
    {
      "event_id": "evt-sim-agent-sdk",
      "title": "Fixture: open-source coding-agent SDK released",
      "category": "Developer Tools / Agent Frameworks",
      "primary_url": "https://github.com/example-labs/fixture-agent-sdk",
      "supporting_urls": [],
      "importance": 8,
      "relevance": 9,
      "actionability": 9,
      "source_quality": 8,
      "confidence": 85,
      "tldr": "An Apache-2.0 Python SDK bundling tool calling, MCP, sandboxing, and evals for coding agents.",
      "what_happened": "Fixture Agent SDK 1.0 released on GitHub.",
      "what_changed": "Previously these pieces had to be assembled by hand.",
      "why_it_matters": "A practical starting point for the reader's own agent projects.",
      "key_takeaways": ["Ships with an evaluation harness"],
      "technical_architecture": ["Tool registry + MCP client + sandboxed runner"],
      "technical_details": ["Python 3.11+", "Apache-2.0"],
      "potential_impact": "Lowers the barrier to building production-grade agents.",
      "action_type": "TRY",
      "action": "Install the SDK and run its example agent locally."
    }
  ],
  "trends": ["Agent memory is converging on learned organization"],
  "strategic_implications": ["Memory design is becoming a core skill"],
  "build_ideas": ["Relation-classifier over toy memories"],
  "learn_next": ["ROAM++ relation taxonomy"],
  "opportunities": ["Reproduce ROAM++ baselines for a blog post"],
  "things_to_ignore": ["Weekend hype threads"]
}
"""


def run_simulated_pipeline(state: dict | None = None) -> bool:
    """Offline end-to-end simulation of the pipeline.

    Uses fixture sources and a fixture Gemini report, then runs the
    REAL curation, validation, publishing, and state-update code.
    No network, no API keys required.
    """
    state = state if state is not None else {}

    logger.info("=== SIMULATION MODE — no network, no API calls ===")

    # --- Collect (fixture) ---
    raw_results = [dict(c) for c in SIMULATED_SOURCES]
    logger.info(f"Step 1/5: Collected {len(raw_results)} simulated raw results")

    # --- Deduplicate (real code; must collapse the arXiv duplicate) ---
    unique_results = collapse_by_dedup_key(deduplicate_results(raw_results))
    logger.info(f"Step 2/5: After dedup: {len(unique_results)} unique sources")

    # --- Filter (real code; enforces seen URLs + dedup keys) ---
    candidates = prepare_candidates(unique_results, state)
    if not candidates:
        logger.info("Step 3/5: No new candidates (all already seen) — nothing to do")
        return False

    # --- Analyze (fixture report instead of Gemini) ---
    report = parse_report(SIMULATED_REPORT_JSON)
    logger.info(f"Step 4/5: Simulated Gemini report: {len(report.events)} events")

    # --- Validate (real code) ---
    report = validate_report(
        report, candidates, state,
        extra_allowed_urls=_collapsed_extra_urls(candidates),
    )
    logger.info(f"Step 4/5: Validated events: {len(report.events)}")

    if not report.events:
        logger.info("No sufficiently important events found")
        return False

    # --- Publish (real code, real files on disk) ---
    outputs = publish_report(report)
    for fmt, path in outputs.items():
        logger.info(f"  Published {fmt}: {path}")

    # --- Update state (publishing counts as delivery in simulation) ---
    update_state(state, report, candidates)
    state["last_report_files"] = outputs
    logger.info("Step 5/5: State updated")

    logger.info("✅ SIMULATION COMPLETE")
    return True
