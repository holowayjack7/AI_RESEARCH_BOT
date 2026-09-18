"""Telegram delivery module.

Formats research reports as strict-structure Telegram HTML messages
and sends them to a configured chat.

Formatting contract:
- Every event follows the mandatory 4-section structure:
  Executive Impact / Key Technical Breakdown / Actionable Takeaways /
  Verified Resources
- Zero emojis, icons, or visual badges anywhere in the output
- Mobile-first: compact cards, collapsed background context behind
  expandable blockquotes, monospace-free clean sections
- Robust: chunking splits at paragraph boundaries (never mid-tag), a
  bad-HTML chunk falls back to plain text, and one failing chunk never
  aborts the whole delivery. Transient Telegram rate limits (429 with
  retry_after) are waited out with up to 5 attempts per chunk.
"""

import logging
import re
import time
from urllib.parse import urlparse

from config import TELEGRAM_CHUNK_SIZE
from src.net import http_post

logger = logging.getLogger("ai_research_bot")

TELEGRAM_HARD_LIMIT = 4096
# Deep per-chunk retries: transient rate limits must not cost us a
# chunk of the user's report.
MAX_SEND_ATTEMPTS = 5

# Inter-chunk pause: Telegram allows ~1 msg/sec per chat
CHUNK_PAUSE_SECONDS = 1.1

DIVIDER = "-------------------------"

SECTION_TITLES = {
    "trends": "TRENDS",
    "strategic_implications": "STRATEGIC IMPLICATIONS",
    "build_ideas": "BUILD IDEAS",
    "learn_next": "LEARN NEXT",
    "opportunities": "OPPORTUNITIES",
    "things_to_ignore": "IGNORE / LOW VALUE",
}

def escape_html(text) -> str:
    """Escape special HTML characters for Telegram."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def expandable_blockquote(text: str) -> str:
    """Wrap text in an expandable blockquote (collapsed by default)."""
    return f"<blockquote expandable>{text}</blockquote>"


def _impact_score(event) -> float:
    """Weighted composite score (0-10): importance dominates."""
    return (
        0.35 * event.importance
        + 0.30 * event.relevance
        + 0.20 * event.actionability
        + 0.15 * event.source_quality
    )


def _resource_label(url: str) -> str:
    """Short descriptive label for a verified-resource link."""
    low = (url or "").lower()
    if "arxiv.org" in low or "huggingface.co/papers" in low:
        return "Research paper"
    if "github.com" in low:
        return "Source code"
    if "docs." in low or "/docs" in low:
        return "Documentation"
    if "releases" in low or "/blog" in low or "changelog" in low:
        return "Release notes"
    domain = urlparse(url).netloc.replace("www.", "")
    return f"Official source ({domain})" if domain else "Official source"


def _clip(text: str, limit: int) -> str:
    text = escape_html(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


# ============================================================
# EVENT CARD (strict 4-section structure, zero emojis)
# ============================================================

def format_event(index: int, event) -> str:
    """Format a single research event as a strict-structure HTML card.

    Mandatory structure — no emojis, icons, or badges anywhere:
    1. Executive Impact
    2. Key Technical Breakdown (Architecture and Specs + Implementation Logic)
    3. Actionable Takeaways (application + commercial potential)
    4. Verified Resources (clean Markdown-style links)
    """
    title = escape_html(event.title)
    category = escape_html(event.category or "Uncategorized")
    classes = " / ".join(event.classification) if getattr(event, "classification", None) else ""
    classification = f" · {escape_html(classes)}" if classes else ""

    blocks = []

    # Header: index + title + quantified context line
    blocks.append(
        f"<b>{index}. {title}</b>\n"
        f"<i>{category}{classification} · Impact {_impact_score(event):.1f}/10 · "
        f"Confidence {event.confidence}%</i>"
    )

    # --- 1. Executive Impact ---
    executive = []
    if event.tldr:
        executive.append(escape_html(event.tldr))
    context = []
    if event.what_happened:
        context.append(f"What happened: {escape_html(event.what_happened)}")
    if event.what_changed:
        context.append(f"What changed: {escape_html(event.what_changed)}")
    if event.why_it_matters:
        context.append(f"Why it matters: {escape_html(event.why_it_matters)}")
    # Fact vs interpretation and what-could-be-wrong keep the critical
    # analysis visible while staying collapsed on mobile.
    if getattr(event, "interpretation", None):
        context.append(
            f"Fact vs interpretation: {escape_html(event.interpretation)}"
        )
    if getattr(event, "counter_argument", None):
        context.append(
            f"What could be wrong: {escape_html(event.counter_argument)}"
        )
    if context:
        executive.append(expandable_blockquote("\n".join(context)))
    if executive:
        blocks.append("\n".join(["<b>Executive Impact</b>"] + executive))

    # --- 2. Key Technical Breakdown ---
    tech = []
    if event.technical_architecture:
        tech.append("<b>Architecture and Specs</b>")
        tech.extend(f"- {escape_html(item)}" for item in event.technical_architecture)
    if event.technical_details:
        if tech:
            tech.append("")
        tech.append("<b>Implementation Logic</b>")
        tech.extend(f"- {escape_html(item)}" for item in event.technical_details)
    if tech:
        blocks.append("\n".join(["<b>Key Technical Breakdown</b>"] + tech))

    # --- 3. Actionable Takeaways ---
    takeaways = []
    if event.key_takeaways:
        takeaways.extend(f"- {escape_html(item)}" for item in event.key_takeaways)
    action_type = (event.action_type or "TRACK").upper()
    if event.action:
        takeaways.append(
            f"- <b>Apply ({action_type})</b>: {escape_html(event.action)}"
        )
    if event.potential_impact:
        takeaways.append(
            f"- <b>Commercial potential</b>: {escape_html(event.potential_impact)}"
        )
    if takeaways:
        blocks.append("\n".join(["<b>Actionable Takeaways</b>"] + takeaways))

    # --- 4. Verified Resources ---
    resources = [
        f'- <a href="{escape_html(event.primary_url)}">'
        f"{_resource_label(event.primary_url)}</a>"
    ]
    for url in event.supporting_urls[:4]:
        resources.append(f'- <a href="{escape_html(url)}">{_resource_label(url)}</a>')
    blocks.append("\n".join(["<b>Verified Resources</b>"] + resources))

    return "\n\n".join(blocks)


# ============================================================
# FULL MESSAGE
# ============================================================

def _list_section(report, key: str, numbered: bool = False) -> str | None:
    """Render one report-level list section as HTML, or None if empty."""
    items = getattr(report, key, None)
    if not items:
        return None

    title = SECTION_TITLES.get(key, key.replace("_", " ").upper())

    lines = [f"<b>{title}</b>"]
    if numbered:
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. {escape_html(item)}")
    else:
        for item in items:
            lines.append(f"- {escape_html(item)}")

    return "\n".join(lines)


def build_telegram_blocks(report) -> list[str]:
    """Build the report as a list of self-contained HTML blocks.

    Each block is a whole header/summary/event/section — never a
    partial event — so chunking can safely group whole blocks.
    """
    blocks = []

    # --- Header card ---
    header = [
        "<b>AI INTELLIGENCE REPORT</b>",
        f"{_today()} · <b>{len(report.events)}</b> curated "
        f"event{'s' if len(report.events) != 1 else ''}",
    ]

    if report.executive_summary:
        header.append(expandable_blockquote(escape_html(report.executive_summary)))

    # In-this-issue index: the whole report scannable in seconds
    if len(report.events) > 1:
        header.append("<b>In this issue</b>")
        for i, event in enumerate(report.events, start=1):
            header.append(f"  {i}. {_clip(event.title, 72)}")

    header.extend(["", DIVIDER, ""])
    blocks.append("\n".join(header))

    # --- Event cards ---
    for index, event in enumerate(report.events, start=1):
        blocks.append(format_event(index, event))

    # --- Report-level sections ---
    sections = [
        _list_section(report, "trends"),
        _list_section(report, "strategic_implications"),
        _list_section(report, "build_ideas"),
        _list_section(report, "learn_next", numbered=True),
        _list_section(report, "opportunities"),
        _list_section(report, "things_to_ignore"),
    ]
    rendered = [s for s in sections if s]
    if rendered:
        blocks.append(DIVIDER)
        blocks.extend(rendered)

    blocks.append(
        f"<i>AI Research Bot · {_today()} · "
        f"{len(report.events)} events</i>"
    )

    return blocks


def build_telegram_message(report) -> str:
    """Convert a ResearchReport into a single Telegram HTML message."""
    return "\n\n".join(build_telegram_blocks(report))


# ============================================================
# DAILY HEARTBEAT (no-news days)
# ============================================================

def build_no_news_message(state: dict | None = None, candidates_reviewed: int = 0) -> str:
    """Build the 'nothing new today' heartbeat message.

    The user expects a daily report; this keeps the daily ritual and
    proves the bot is alive even when no event passes the filters.
    """
    state = state or {}

    if candidates_reviewed > 0:
        reason = (
            f"{candidates_reviewed} candidate{'s were' if candidates_reviewed != 1 else ' was'} "
            "reviewed today — none met the quality thresholds "
            "(importance / relevance / confidence)."
        )
    else:
        reason = (
            "All monitored sources were scanned — nothing new surfaced "
            "(everything already seen or below the quality bar)."
        )

    tracked_events = len(state.get("events", []))
    tracked_topics = len(state.get("topics", {}))

    lines = [
        "<b>AI INTELLIGENCE REPORT</b>",
        f"<b>No new intelligence today</b>",
        f"{_today()}",
        "",
        expandable_blockquote(escape_html(reason)),
        "",
        DIVIDER,
        "",
        "Sources: arXiv · Hugging Face Papers · web search",
        f"Memory: {tracked_events} delivered events across "
        f"{tracked_topics} topic{'s' if tracked_topics != 1 else ''}",
        "",
        "<i>AI Research Bot · daily monitor — next check tomorrow</i>",
    ]
    return "\n".join(lines)


def build_delayed_message(reason: str) -> str:
    """Build the 'report delayed' notice for crashed runs.

    The user expects a daily message: if the pipeline dies before
    analysis completes (e.g. the model provider is overloaded), send
    a short notice instead of silence.
    """
    return "\n".join([
        "<b>AI INTELLIGENCE REPORT</b>",
        "<b>Today's report is delayed</b>",
        f"{_today()}",
        "",
        expandable_blockquote(_clip(reason, 300)),
        "",
        DIVIDER,
        "",
        "<i>The run will retry automatically tomorrow — or trigger "
        "it now from GitHub Actions.</i>",
    ])


# ============================================================
# CHUNKING
# ============================================================

def split_message(message: str, max_length: int = TELEGRAM_CHUNK_SIZE) -> list[str]:
    """Split a message into Telegram-compatible chunks.

    Splits at paragraph boundaries (blank lines), so HTML tags never
    break mid-tag and every chunk is valid Telegram HTML. A paragraph
    longer than max_length is clipped with any open tags closed.
    """
    chunks = _split_blocks(message.split("\n\n"), max_length)

    if chunks and all(len(c) <= max_length for c in chunks):
        return chunks

    # Should not happen (paragraph splitter guarantees limits), but be
    # safe: strip tags so hard slicing can never produce broken HTML
    text = _strip_tags(message)
    return [
        text[i : i + max_length]
        for i in range(0, len(text), max_length)
    ] or [message[:max_length]]


def _split_blocks(paragraphs: list[str], max_length: int) -> list[str]:
    """Group whole paragraphs into chunks under max_length."""
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_length:
            current = paragraph
        else:
            # Oversized single paragraph: clip it (close any tags a
            # mid-clip would leave open)
            chunks.append(_safe_clip(paragraph, max_length))
            current = ""

    if current:
        chunks.append(current)

    return chunks


# Exact open-tag prefixes -> closing tags. Prefix matching ("<b") would
# wrongly also count "<blockquote", producing unbalanced HTML.
_OPEN_CLOSE_TAGS = (
    ("<b>", "</b>"),
    ("<i>", "</i>"),
    ("<code>", "</code>"),
    ("<a ", "</a>"),
    ("<blockquote ", "</blockquote>"),
)


def _safe_clip(paragraph: str, max_length: int) -> str:
    """Clip a paragraph to max_length, closing any unclosed inline tags."""
    clipped = paragraph[:max_length]
    # Drop a tag truncated mid-way by the cut (e.g. "...<bloc") — a
    # dangling "<" would break Telegram's HTML parser
    clipped = re.sub(r"<[^>]*$", "", clipped)
    # Close tags that the cut could have left open
    for open_tag, close_tag in _OPEN_CLOSE_TAGS:
        if clipped.count(open_tag) > clipped.count(close_tag):
            clipped += close_tag
    return clipped


# ============================================================
# SENDING
# ============================================================

def get_telegram_me(bot_token: str) -> dict:
    """Verify a bot token via getMe and return the bot info.

    Raises on invalid token or network failure — used to diagnose
    delivery problems before a full pipeline run.
    """
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    response = http_post(url, {}, timeout=15, max_attempts=2)
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"getMe failed: {data.get('description', data)}")
    return data.get("result", {})


def send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """Send a message to Telegram, splitting into chunks if needed.

    Resilience (the report MUST reach the chat):
    - Each chunk gets up to MAX_SEND_ATTEMPTS at the HTTP layer, where
      transient errors (429/5xx/network) are waited out with backoff,
      honoring Telegram's retry_after.
    - A chunk that permanently fails (e.g. HTML parse error) is retried
      once as plain text.
    - If that also fails, the chunk is skipped and delivery continues.
    - Permanent account-level errors (401/403: bot blocked or removed,
      chat not found) abort delivery immediately — retrying other
      chunks to the same chat can never succeed.
    - Returns True if at least one chunk was delivered.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    chunks = split_message(message)
    success = False

    for index, chunk in enumerate(chunks, start=1):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "link_preview_options": {"is_disabled": True},
        }

        try:
            response = http_post(
                url, payload, timeout=20, max_attempts=MAX_SEND_ATTEMPTS
            )
            logger.info(
                f"Telegram chunk {index}/{len(chunks)}: {response.status_code}"
            )
            success = True

        except Exception as e:
            # Account-level permanent failure: abort the whole send —
            # remaining chunks to the same chat will fail identically
            if _is_permanent_send_error(str(e)):
                logger.error(
                    f"Telegram delivery aborted: permanent error ({e}) — "
                    f"check that the bot is not blocked and the chat id "
                    f"is correct"
                )
                return False
            # Retry once without HTML parsing — the most common
            # permanent failure is a Telegram parse error
            logger.warning(
                f"Telegram chunk {index}/{len(chunks)} failed ({e}) — "
                f"retrying as plain text"
            )
            try:
                fallback = {
                    "chat_id": chat_id,
                    "text": _strip_tags(chunk)[: TELEGRAM_HARD_LIMIT - 100],
                    "disable_web_page_preview": True,
                }
                http_post(
                    url, fallback, timeout=20, max_attempts=MAX_SEND_ATTEMPTS
                )
                success = True
            except Exception as e2:
                if _is_permanent_send_error(str(e2)):
                    logger.error(
                        f"Telegram delivery aborted: permanent error "
                        f"({e2}) — check that the bot is not blocked and "
                        f"the chat id is correct"
                    )
                    return False
                logger.error(
                    f"Telegram chunk {index}/{len(chunks)} permanently "
                    f"failed: {e2} — continuing with remaining chunks"
                )

        # Stay friendly to Telegram's ~1 msg/sec per-chat limit
        if index < len(chunks):
            time.sleep(CHUNK_PAUSE_SECONDS)

    return success


_PERMANENT_SEND_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "blocked by the user",
    "bot was blocked",
    "chat not found",
    "bot was kicked",
)


def _is_permanent_send_error(error_str: str) -> bool:
    """True for Telegram errors where retrying other chunks is futile.

    A blocked bot, revoked token, or wrong chat id fails for every
    chunk identically — fail fast with a clear log instead of grinding
    through the whole report.
    """
    s = (error_str or "").lower()
    return any(marker in s for marker in _PERMANENT_SEND_MARKERS)


def _strip_tags(text: str) -> str:
    """Remove HTML tags for plain-text fallback."""
    return re.sub(r"<[^>]+>", "", text)
