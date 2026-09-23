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
    GEMINI_MODEL,
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
    # Broad AI news outlets (reputable, still signal-filtered downstream)
    "techcrunch.com": 7, "theverge.com": 7, "venturebeat.com": 7,
    "arstechnica.com": 8, "wired.com": 7, "engadget.com": 7,
    "reuters.com": 8,
    # High-signal AI engineering voices/blogs
    "simonwillison.net": 9, "interconnects.ai": 8, "latent.space": 8,
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
# DECISION RULES (code-level intelligence gate)
# ============================================================

# Classifications considered "hype-adjacent": they must clear a
# higher bar, because they describe attention, not durable knowledge.
HYPE_CLASSIFICATIONS = {"temporary trend", "general news"}
# Classifications that can rescue clearly-labeled speculation: labeled
# foresight with real strategic value survives the factuality gate.
# Includes the opportunity horizons — a clearly-labeled short/long-term
# opportunity is exactly the strategic speculation worth keeping.
STRATEGIC_CLASSIFICATIONS = {
    "real business opportunity",
    "long-term career value",
    "short-term opportunity",
    "long-term opportunity",
}

# Composite quality bar (importance alone never decides). Weights:
# evidence/confidence and relevance dominate; actionability and source
# quality support. Sum of weights = 1.0 -> composite is 0-10.
COMPOSITE_WEIGHTS = {
    "importance": 0.25,
    "relevance": 0.25,
    "actionability": 0.15,
    "confidence": 0.25,   # (confidence / 10) -> 0-10 scale
    "source_quality": 0.10,
}
MIN_COMPOSITE_SCORE = 6.0

# Minimum importance for hype-adjacent classifications
MIN_HYPE_IMPORTANCE = 8


def composite_score(event) -> float:
    """Weighted multi-factor quality score (0-10) for an event.

    The final send decision must consider evidence quality, relevance,
    practical value, and source quality — never the importance score
    alone.
    """
    return (
        COMPOSITE_WEIGHTS["importance"] * event.importance
        + COMPOSITE_WEIGHTS["relevance"] * event.relevance
        + COMPOSITE_WEIGHTS["actionability"] * event.actionability
        + COMPOSITE_WEIGHTS["confidence"] * (event.confidence / 10.0)
        + COMPOSITE_WEIGHTS["source_quality"] * event.source_quality
    )


def _norm_classifications(event) -> set[str]:
    """Lower-cased set of the event's classification labels."""
    return {c.strip().lower() for c in (event.classification or []) if c.strip()}


def passes_decision_rules(
    event,
    analysis_by_url: dict | None = None,
) -> bool:
    """Apply the critical-thinking decision rules to one event.

    Returns True if the event should be reported. Enforces, in order:
    1. per-article gate — the underlying article must be relevant and
       marked should_send by the pre-selection analysis;
    2. factuality gate — speculative content is rejected UNLESS it is
       clearly classified as a strategic category (labeled speculation
       with meaningful insight survives);
    3. composite multi-factor gate — importance, relevance,
       actionability, confidence, and source quality together must
       clear MIN_COMPOSITE_SCORE (the importance score alone never
       decides);
    4. hype deprioritization — Temporary Trend / General News need a
       higher importance to justify the reader's attention.
    """
    # --- 1. Per-article gate ---
    if analysis_by_url is not None:
        analysis = analysis_by_url.get(event.primary_url)
        if analysis is not None:
            if not analysis.get("is_relevant", True):
                logger.info(
                    f"Rejected (underlying article not relevant): "
                    f"{event.primary_url}"
                )
                return False
            if not analysis.get("should_send", True):
                logger.info(
                    f"Rejected (analysis marked should_send=false): "
                    f"{event.primary_url}"
                )
                return False

    # --- 2. Factuality gate ---
    classifications = _norm_classifications(event)
    if event.factuality_level == "speculative" and not (
        classifications & STRATEGIC_CLASSIFICATIONS
    ):
        logger.info(
            f"Rejected (unlabeled speculation without strategic value): "
            f"{event.title[:60]}"
        )
        return False

    # --- 3. Composite multi-factor gate ---
    score = composite_score(event)
    if score < MIN_COMPOSITE_SCORE:
        logger.info(
            f"Rejected (composite {score:.2f} < {MIN_COMPOSITE_SCORE}): "
            f"{event.title[:60]}"
        )
        return False

    # --- 4. Hype deprioritization ---
    if classifications & HYPE_CLASSIFICATIONS and event.importance < MIN_HYPE_IMPORTANCE:
        logger.info(
            f"Rejected (hype-classified but importance {event.importance} "
            f"< {MIN_HYPE_IMPORTANCE}): {event.title[:60]}"
        )
        return False

    return True


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

    # Per-article critical analyses: index the pre-selection verdicts
    # by URL so each event inherits its article's decision.
    analysis_by_url: dict = {}
    for analysis in getattr(report, "candidate_analyses", []) or []:
        if getattr(analysis, "url", ""):
            analysis_by_url[analysis.url] = {
                "is_relevant": analysis.is_relevant,
                "should_send": analysis.should_send,
            }

    valid_events = []
    deprioritized = []

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

        # Critical-thinking decision rules (per-article gate,
        # factuality gate, composite score, hype deprioritization)
        if not passes_decision_rules(event, analysis_by_url):
            continue

        # Durable knowledge first: hype-adjacent classifications sort
        # behind strategic/technical ones at equal importance.
        if _norm_classifications(event) & HYPE_CLASSIFICATIONS:
            deprioritized.append(event)
        else:
            valid_events.append(event)

    # Sort by importance, relevance, actionability, confidence;
    # hype-adjacent events follow the durable ones.
    rank_key = lambda e: (e.importance, e.relevance, e.actionability, e.confidence)
    valid_events.sort(key=rank_key, reverse=True)
    deprioritized.sort(key=rank_key, reverse=True)
    valid_events.extend(deprioritized)

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
    gemini_model: str = GEMINI_MODEL,
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
  "report_title": "სიმულაცია — AI დაზვერვის რეპორტი",
  "executive_summary": "სიმულაციური რანი ამოწმებს offline pipeline-ს: curation, validation, export და state tracking.",
  "events": [
    {
      "event_id": "evt-sim-roam-plus",
      "title": "ROAM++: Self-Organizing Agent Memory via Learned Relations",
      "category": "AI კვლევა / Agent Memory",
      "primary_url": "https://arxiv.org/abs/2609.12001",
      "supporting_urls": ["https://huggingface.co/papers/2609.12001"],
      "importance": 9,
      "relevance": 10,
      "actionability": 8,
      "source_quality": 8,
      "confidence": 90,
      "tldr": "Agent memory, რომელიც სწავლობს მეხსიერებებს შორის კავშირებს: +34.2pp სიზუსტე, retrieval latency 22%-ით დაბალი.",
      "what_happened": "გამოქვეყნდა ROAM++ — კოდით, benchmark-ებით და evaluation harness-ებით.",
      "what_changed": "Memory ორგანიზაცია: ფიქსირებული წესები -> ნასწავლი semantic relations.",
      "why_it_matters": "პირდაპირ ავითარებს agent-memory roadmap-ს; დღესვე გამეორებადია.",
      "key_takeaways": [
        "ნასწავლი relations ამარცხებს fixed heuristics-ს",
        "+34.2pp compaction baselines-თან შედარებით",
        "კოდი და eval-ები საჯაროა"
      ],
      "technical_architecture": [
        "Atomic memories + learned relation graph",
        "Deterministic reorganization after inference"
      ],
      "technical_details": ["LoCoMo +34.2pp", "-22% retrieval latency"],
      "potential_impact": "ცვლის design baseline-ს გრძელვადიანი agent memory-სთვის.",
      "action_type": "LEARN",
      "action": "წაიკითხე paper; დახატე relation-classifier-ის სქემა toy memories-ზე.",
      "factuality_level": "verified",
      "classification": ["Real Technical Skill", "Long-Term Career Value"],
      "verified_facts": [
        "+34.2pp LoCoMo-ზე compaction baselines-თან შედარებით",
        "retrieval latency 22%-ით ნაკლები",
        "კოდი და eval-ები საჯაროა"
      ],
      "interpretation": "Memory ორგანიზაცია გადადის hand-tuned policies-დან ნასწავლებზე.",
      "uncertainty": "Benchmark-ები შეიძლება ამჯობინებდეს ავტორთა საკუთარ design choices-ს.",
      "counter_argument": "მოგება შეიძლება შემცირდეს აკადემიურ გარემოში, curation-ის გარეშე.",
      "strategic_assessment": "რეკომენდაცია: აითვისე learned memory organization ახლავე — ეს 3-10 წლის horizon-ზე agent engineering-ის საფუძველი გახდება.",
      "time_horizon": "long-term",
      "what_could_change": "თუ retrieval-ზე ფასი კიდევ დაეცემა, long-context models პირდაპირ კონკურენციას გაუწევს external memory-ს.",
      "risks": ["Autori benchmark-ები შეიძლება cherry-picked იყოს", "Production workloads-ზე შედეგები შეიძლება არ გადავიდეს"],
      "non_obvious_opportunity": "Relation taxonomy შეიძლება გამოყენებულ იქნას tool-calling graphs-ზეც — არა მხოლოდ memory-ზე.",
      "fit_assessment": "იდეალური ფიტი: Python + agent loops roadmap-ის ზუსტად შემდეგი ნაბიჯი, უფასოა და portfolio-ს ძლიერი ნაწილი გახდება."
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
      "tldr": "Apache-2.0 Python SDK, რომელიც აერთიანებს tool calling-ს, MCP-ს, sandbox-ს და eval-ებს.",
      "what_happened": "Fixture Agent SDK 1.0 გამოქვეყნდა GitHub-ზე.",
      "what_changed": "ხელით აწყობილი agent plumbing -> ერთი installable SDK.",
      "why_it_matters": "ყველაზე სწრაფი გზა coding-agent prototype-ამდე ამ კვირაში.",
      "key_takeaways": [
        "მოყვება evaluation harness",
        "MCP client ჩაშენებულია"
      ],
      "technical_architecture": [
        "Tool registry + MCP client + sandboxed runner"
      ],
      "technical_details": ["Python 3.11+", "Apache-2.0"],
      "potential_impact": "coding-agent bootstrap დროს აკლებს დღეებიდან საათებამდე.",
      "action_type": "TRY",
      "action": "დააინსტალირე SDK; გაუშვი მისი example agent ლოკალურად.",
      "factuality_level": "corroborated",
      "classification": ["Real Technical Skill", "Real Business Opportunity"],
      "verified_facts": [
        "Apache-2.0 Python SDK",
        "Tool calling, MCP client, sandboxed execution შედის"
      ],
      "interpretation": "Agent plumbing კონსოლიდირდება installable SDK-ებში.",
      "uncertainty": "მოვლა და გრძელვადიანი მხარდაჭერა უცნობია.",
      "counter_argument": "Lock-in risk: SDK-ის abstractions-მა შეიძლება MCP-დან ჩამოცილდეს.",
      "strategic_assessment": "გამოიყენე prototype-ებისთვის, მაგრამ core abstractions-ები საკუთარი დაწერე — სწავლა ღირებულებაა, არა lock-in.",
      "time_horizon": "short-term",
      "what_could_change": "თუ major vendor ჩაშენებს მსგავს ფუნქციას თავის API-ში, standalone SDK-ების ღირებულება დაეცემა.",
      "risks": ["ადრეული SDK-ები ხშირად იცვლის API-ს", "Maintenance შეიძლება შეწყდეს"],
      "non_obvious_opportunity": "SDK-ის eval harness შეიძლება გამოყენებულ იქნას საკუთარი agents-ის შესაფასებლად, SDK-ის გარეშე.",
      "fit_assessment": "კარგი ფიტი კვირის prototype-სთვის; გრძელვადიან skill-ად საკუთარი implementation უფრო ღირებულია."
    }
  ],
  "trends": ["Agent memory იწყებს კონვერგენციას learned organization-ზე"],
  "strategic_implications": ["Memory design ხდება ძირითადი skill"],
  "build_ideas": ["Relation-classifier toy memories-ზე"],
  "learn_next": ["ROAM++ relation taxonomy"],
  "opportunities": ["ROAM++ baselines-ის გამეორება blog post-თვის"],
  "things_to_ignore": ["უქვეითენდო hype thread-ები"],
  "action_plan": {
    "today": ["გადაწერე ROAM++-ის relation classifier-ის სქემა ქაღალდზე", "დააინსტალირე fixture SDK და გაუშვი მისი example agent"],
    "this_week": ["ააწყოს minimal agent memory toy 10 test question-ზე", "გაზომე retrieval accuracy baseline-ის წინააღმდეგ", "დაწერე SDK-ის sandbox execution-ის მიმოხილვა blog-ისთვის"],
    "next": ["შეისწავლე relation taxonomy და გაავრცხეlle tool-calling graphs-ზე", "ააწყოს self-evaluation loop საკუთარი agent-ისთვის"],
    "stop_ignore": ["multi-agent framework hype thread-ები", "ახალი model releases, რომლებსაც benchmark-ები არ მოჰყვება"]
  },
  "candidate_analyses": [
    {
      "title": "ROAM++: Self-Organizing Agent Memory via Learned Relations",
      "url": "https://arxiv.org/abs/2609.12001",
      "is_relevant": true,
      "source_quality": 8,
      "factuality_level": "verified",
      "importance_score": 9,
      "classification": ["Real Technical Skill", "Long-Term Career Value"],
      "verified_facts": [
        "+34.2pp LoCoMo-ზე",
        "-22% retrieval latency",
        "კოდი გამოქვეყნებულია"
      ],
      "interpretation": "ნასწავლი memory ორგანიზაცია default გახდება.",
      "uncertainty": "ერთი გუნდის benchmark design.",
      "counter_argument": "აკადემიური workloads შეიძლება production agents-ზე არ გადავიდეს.",
      "actionable_takeaway": "შეისწავლე relation taxonomy შემდეგ memory design-მდე.",
      "should_send": true
    },
    {
      "title": "Fixture: open-source coding-agent SDK released",
      "url": "https://github.com/example-labs/fixture-agent-sdk",
      "is_relevant": true,
      "source_quality": 8,
      "factuality_level": "corroborated",
      "importance_score": 8,
      "classification": ["Real Technical Skill", "Real Business Opportunity"],
      "verified_facts": ["Apache-2.0", "MCP client შედის"],
      "interpretation": "SDK კონსოლიდაცია ამცირებს agent development-ის ბარიერს.",
      "uncertainty": "პროექტის maturity უცნობია.",
      "counter_argument": "ადრეული SDK-ები ხშირად სწრაფად ტოვებენ abstractions-ს.",
      "actionable_takeaway": "ამ კვირაში ააწყვე ერთი tool-serving agent prototype.",
      "should_send": true
    }
  ]
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
