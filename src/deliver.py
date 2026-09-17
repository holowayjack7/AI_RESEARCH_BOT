"""Telegram delivery module.

Formats research reports as modern, readable Telegram HTML messages
and sends them to a configured chat.

UI principles (Telegram 2024+ features):
- Expandable blockquotes keep cards compact: long context is collapsed
  behind a tap (executive summary, event background)
- Scannable: medal ranks for the top events, category chip, source-type
  chip, one-line TL;DR first
- Quantified: a compact monospace score strip plus a weighted impact
  score per event
- Actionable: every event ends with a highlighted action callout
- Robust: chunking splits at paragraph boundaries (never mid-tag), a
  bad-HTML chunk falls back to plain text, and one failing chunk never
  aborts the whole delivery. Transient Telegram rate limits (429 with
  retry_after) are waited out with up to 5 attempts per chunk.
"""

import logging
import re
import time

from config import TELEGRAM_CHUNK_SIZE
from src.net import http_post

logger = logging.getLogger("ai_research_bot")

TELEGRAM_HARD_LIMIT = 4096
# Deep per-chunk retries: transient rate limits must not cost us a
# chunk of the user's report.
MAX_SEND_ATTEMPTS = 5

# Inter-chunk pause: Telegram allows ~1 msg/sec per chat
CHUNK_PAUSE_SECONDS = 1.1

ACTION_EMOJI = {
    "BUILD": "🛠",
    "TRY": "🧪",
    "LEARN": "📚",
    "TRACK": "👀",
    "APPLY": "⚙️",
    "IGNORE": "🚫",
}

SECTION_EMOJI = {
    "trends": "📈",
    "strategic_implications": "🧭",
    "build_ideas": "🛠",
    "learn_next": "📚",
    "opportunities": "🎯",
    "things_to_ignore": "🚫",
}

SECTION_TITLES = {
    "trends": "TRENDS",
    "strategic_implications": "STRATEGIC IMPLICATIONS",
    "build_ideas": "BUILD IDEAS",
    "learn_next": "LEARN NEXT",
    "opportunities": "OPPORTUNITIES",
    "things_to_ignore": "IGNORE / LOW VALUE",
}

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


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


def _score_bar(value: int, max_value: int = 10) -> str:
    """Render a value as a filled/empty bar, e.g. █████████░ 9/10."""
    filled = max(0, min(max_value, int(value)))
    return "█" * filled + "░" * (max_value - filled)


def _impact_score(event) -> float:
    """Weighted composite score (0-10): importance dominates."""
    return (
        0.35 * event.importance
        + 0.30 * event.relevance
        + 0.20 * event.actionability
        + 0.15 * event.source_quality
    )


def _source_chip(url: str) -> str:
    """A small source-type emoji derived from the URL domain."""
    url = (url or "").lower()
    if "arxiv.org" in url or "huggingface.co/papers" in url:
        return "📄 paper"
    if "github.com" in url:
        return "🛠 code"
    return "📰 news"


def _clip(text: str, limit: int) -> str:
    text = escape_html(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


# ============================================================
# EVENT CARD
# ============================================================

def format_event(index: int, event) -> str:
    """Format a single research event as a modern Telegram HTML card."""
    title = escape_html(event.title)
    category = escape_html(event.category or "Uncategorized")
    rank = MEDALS.get(index, "")
    rank_prefix = f"{rank} " if rank else ""

    lines = []

    # Header: rank + title + source-type chip
    lines.append(
        f"<b>{rank_prefix}{index}. {title}</b>\n"
        f"┋ 🏷 <i>{category}</i> · {_source_chip(event.primary_url)} · "
        f"✅ {event.confidence}%"
    )

    # Compact score strip (monospace, aligned) + weighted impact score
    lines.append(
        "<code>"
        f"I {_score_bar(event.importance)} {event.importance}/10\n"
        f"R {_score_bar(event.relevance)} {event.relevance}/10\n"
        f"A {_score_bar(event.actionability)} {event.actionability}/10\n"
        f"S {_score_bar(event.source_quality)} {event.source_quality}/10"
        "</code>\n"
        f"<i><code>  I·importance R·relevance A·action S·source "
        f"— ◆ impact {_impact_score(event):.1f}/10</code></i>"
    )

    # TL;DR first — one dense sentence
    if event.tldr:
        lines.append(f"⚡ <b>TL;DR</b> — {escape_html(event.tldr)}")

    # Background context (what happened / changed / why) — collapsed
    # behind an expandable blockquote to keep the feed scannable
    context = []
    if event.what_happened:
        context.append(f"<b>What happened.</b> {escape_html(event.what_happened)}")
    if event.what_changed:
        context.append(f"<b>What changed.</b> {escape_html(event.what_changed)}")
    if event.why_it_matters:
        context.append(f"<b>Why it matters.</b> {escape_html(event.why_it_matters)}")
    if context:
        lines.append(expandable_blockquote("\n".join(context)))

    if event.key_takeaways:
        lines.append("<b>Key takeaways</b>")
        for takeaway in event.key_takeaways:
            lines.append(f"  ▸ {escape_html(takeaway)}")

    if event.technical_architecture:
        lines.append("<b>Architecture</b>")
        for item in event.technical_architecture:
            lines.append(f"  ◆ {escape_html(item)}")

    if event.technical_details:
        lines.append("<b>Details</b>")
        for detail in event.technical_details:
            lines.append(f"  • {escape_html(detail)}")

    if event.potential_impact:
        lines.append(f"💫 <b>Impact.</b> {escape_html(event.potential_impact)}")

    # Action callout
    action_emoji = ACTION_EMOJI.get((event.action_type or "").upper(), "▶️")
    if event.action:
        lines.append(
            f"{action_emoji} <b>ACTION · {escape_html((event.action_type or '').upper())}</b>\n"
            f"└ {escape_html(event.action)}"
        )

    # Sources
    sources = [
        f'🔗 <a href="{escape_html(event.primary_url)}">primary source</a>'
    ]
    for url in event.supporting_urls[:4]:
        sources.append(f'<a href="{escape_html(url)}">supporting</a>')
    lines.append(" · ".join(sources))

    return "\n".join(lines)


# ============================================================
# FULL MESSAGE
# ============================================================

def _list_section(report, key: str, numbered: bool = False) -> str | None:
    """Render one report-level list section as HTML, or None if empty."""
    items = getattr(report, key, None)
    if not items:
        return None

    emoji = SECTION_EMOJI.get(key, "•")
    title = SECTION_TITLES.get(key, key.replace("_", " ").upper())

    lines = [f"<b>{emoji} {title}</b>"]
    if numbered:
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. {escape_html(item)}")
    else:
        for item in items:
            lines.append(f"▸ {escape_html(item)}")

    return "\n".join(lines)


def build_telegram_blocks(report) -> list[str]:
    """Build the report as a list of self-contained HTML blocks.

    Each block is a whole header/summary/event/section — never a
    partial event — so chunking can safely group whole blocks.
    """
    blocks = []

    # --- Header card ---
    divider = "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈"
    header = [
        "<b>🛰 AI INTELLIGENCE REPORT</b>",
        f"📅 {_today()} · <b>{len(report.events)}</b> curated "
        f"event{'s' if len(report.events) != 1 else ''}",
    ]

    if report.executive_summary:
        header.append(expandable_blockquote(escape_html(report.executive_summary)))

    # In-this-issue index: the whole report scannable in seconds
    if len(report.events) > 1:
        header.append("<b>📋 In this issue</b>")
        for i, event in enumerate(report.events, start=1):
            marker = MEDALS.get(i, f"{i}.")
            header.append(f"  {marker} {_clip(event.title, 72)}")

    header.extend(["", divider, ""])
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
        blocks.append(divider)
        blocks.extend(rendered)

    blocks.append(
        f"<i>🤖 AI Research Bot · {_today()} · "
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
    divider = "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈"

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
        "<b>🛰 AI INTELLIGENCE REPORT</b>",
        f"📭 <b>No new intelligence today</b>",
        f"📅 {_today()}",
        "",
        expandable_blockquote(escape_html(reason)),
        "",
        divider,
        "",
        f"📡 Sources: arXiv · Hugging Face Papers · web search",
        f"📈 Memory: {tracked_events} delivered events across "
        f"{tracked_topics} topic{'s' if tracked_topics != 1 else ''}",
        "",
        "<i>🤖 AI Research Bot · daily monitor — next check tomorrow</i>",
    ]
    return "\n".join(lines)


def build_delayed_message(reason: str) -> str:
    """Build the 'report delayed' notice for crashed runs.

    The user expects a daily message: if the pipeline dies before
    analysis completes (e.g. the model provider is overloaded), send
    a short notice instead of silence.
    """
    divider = "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈"
    return "\n".join([
        "<b>🛰 AI INTELLIGENCE REPORT</b>",
        "⏳ <b>Today's report is delayed</b>",
        f"📅 {_today()}",
        "",
        expandable_blockquote(_clip(reason, 300)),
        "",
        divider,
        "",
        "<i>🤖 The run will retry automatically tomorrow — or trigger "
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
                logger.error(
                    f"Telegram chunk {index}/{len(chunks)} permanently "
                    f"failed: {e2} — continuing with remaining chunks"
                )

        # Stay friendly to Telegram's ~1 msg/sec per-chat limit
        if index < len(chunks):
            time.sleep(CHUNK_PAUSE_SECONDS)

    return success


def _strip_tags(text: str) -> str:
    """Remove HTML tags for plain-text fallback."""
    return re.sub(r"<[^>]+>", "", text)
