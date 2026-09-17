"""LLM analysis module using Gemini.

Analyzes collected research data and produces a structured
intelligence report using Google's Gemini API.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from config import (
    GEMINI_CONTENT_CHARS,
    GEMINI_MAX_ATTEMPTS,
    GEMINI_MODEL_FALLBACKS,
    GEMINI_PROMPT_CHAR_BUDGET,
)
from google import genai
from google.genai import types

logger = logging.getLogger("ai_research_bot")


# ============================================================
# PERSONAL RESEARCH PROFILE
# ============================================================

USER_PROFILE = """
The reader is a 16-year-old developer in Georgia building toward
independent AI / Agentic Engineering.

Current roadmap:
Python → APIs / HTTP / JSON → SQL / Databases → LLM APIs →
Tool Calling / MCP → Agent Loops → State → Graphs → RAG →
Evaluation → Production → Self-improvement → Real Business Agents

Strong interests:
- AI agents, agent architecture, agent loops
- state and memory, MCP, tool calling
- LLM APIs, RAG, evaluation, reliability
- AI coding tools, open-source AI
- AI engineering, practical projects, automation
- AI business opportunities, free/low-cost tools
- opportunities accessible to teenagers 16+

The reader does NOT want generic AI news.

Prioritize information that can:
1. improve engineering ability,
2. change the architecture of future projects,
3. reveal important new AI capabilities,
4. provide something worth building/testing,
5. reveal important ecosystem trends,
6. provide a real learning opportunity,
7. provide a realistic opportunity for a 16+ developer.

Deprioritize:
- stock prices, corporate financial news
- generic AI hype, celebrity/CEO news
- generic image/video generation
- consumer AI features with no engineering value
- minor product updates, repetitive announcements.
"""


# ============================================================
# OUTPUT MODELS (dataclasses for structured Gemini output)
# ============================================================

@dataclass
class Evidence:
    url: str = ""
    title: str = ""
    domain: str = ""
    evidence: str = ""


@dataclass
class ResearchEvent:
    event_id: str = ""
    title: str = ""
    category: str = ""
    primary_url: str = ""
    supporting_urls: list[str] = field(default_factory=list)
    importance: int = 5
    relevance: int = 5
    actionability: int = 5
    source_quality: int = 5
    confidence: int = 50
    tldr: str = ""
    what_happened: str = ""
    what_changed: str = ""
    why_it_matters: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    technical_architecture: list[str] = field(default_factory=list)
    technical_details: list[str] = field(default_factory=list)
    potential_impact: str = ""
    action_type: str = "TRACK"
    action: str = ""
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class ResearchReport:
    report_title: str = ""
    executive_summary: str = ""
    events: list[ResearchEvent] = field(default_factory=list)
    trends: list[str] = field(default_factory=list)
    strategic_implications: list[str] = field(default_factory=list)
    build_ideas: list[str] = field(default_factory=list)
    learn_next: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)
    things_to_ignore: list[str] = field(default_factory=list)


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def _slim_memory_items(items: list) -> list[dict]:
    """Slim stored events for the prompt (drop heavy prose fields).

    Full event dicts (takeaways, architecture, Georgian text, etc.)
    can burn hundreds of thousands of tokens against Gemini's
    per-minute quota; titles + ids + urls are enough for dedup.
    """
    slim = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slim.append({
            "event_id": item.get("event_id", ""),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "primary_url": item.get("primary_url", ""),
        })
    return slim


def _slim_memory_reports(reports: list) -> list[dict]:
    """Slim stored reports for the prompt (summary + titles only)."""
    slim = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        slim.append({
            "date": report.get("date", ""),
            "summary": report.get("summary", ""),
            "event_titles": [
                e.get("title", "")
                for e in report.get("events", [])
                if isinstance(e, dict)
            ],
        })
    return slim


def build_research_input(candidates: list[dict], state: dict | None = None) -> dict:
    """Prepare research data for Gemini analysis.

    Includes recent memory from persistent state so Gemini can
    avoid re-reporting known events and track long-term trends.

    Total prompt size is bounded by GEMINI_PROMPT_CHAR_BUDGET so a
    daily run can never blow through the API's per-minute token
    quota (the failure mode seen in CI: 429 RESOURCE_EXHAUSTED on
    all attempts).
    """
    state = state or {}

    sources = []
    memory_budget = 8000   # cap for slimmed historical memory
    content_budget = max(1000, GEMINI_PROMPT_CHAR_BUDGET - memory_budget)

    for candidate in candidates:
        used = sum(len(s["content"]) for s in sources)
        remaining = content_budget - used
        if remaining <= 200:
            logger.info(
                f"Prompt content budget reached — sending first "
                f"{len(sources)} of {len(candidates)} candidates"
            )
            break
        sources.append({
            "title": candidate["title"],
            "url": candidate["url"],
            "domain": candidate["domain"],
            "source_quality": candidate.get("source_quality", 5),
            "search_score": candidate.get("score", 0),
            "content": (candidate.get("content") or "")[
                : min(GEMINI_CONTENT_CHARS, remaining)
            ],
        })

    historical_events = _slim_memory_items(state.get("events", [])[-40:])
    historical_reports = _slim_memory_reports(state.get("reports", [])[-7:])
    learning_queue = state.get("learning_queue", [])[-20:]

    return {
        "date": time.strftime("%Y-%m-%d"),
        "profile": USER_PROFILE,
        "sources": sources,
        "historical_events": historical_events,
        "historical_reports": historical_reports,
        "learning_queue": learning_queue,
    }


def build_gemini_prompt(research_data: dict) -> str:
    """Build the analysis prompt for Gemini."""
    return f"""
You are the intelligence analyst for a personal AI Engineering research system.

DATE: {research_data["date"]}

USER PROFILE: {research_data["profile"]}

INPUT SOURCES:
{json.dumps(research_data["sources"], ensure_ascii=False, indent=2)}

RECENT MEMORY (events from previous runs):
{json.dumps(research_data["historical_events"], ensure_ascii=False, indent=2)}

RECENT REPORTS:
{json.dumps(research_data["historical_reports"], ensure_ascii=False, indent=2)}

CURRENT LEARNING QUEUE:
{json.dumps(research_data["learning_queue"], ensure_ascii=False, indent=2)}

============================================================
MISSION
============================================================

Produce a DEEP DAILY AI INTELLIGENCE REPORT.

This is NOT a news summary. The objective is to identify information
that can materially improve the reader's AI engineering knowledge,
projects, decisions, or opportunities.

Use the supplied source content as evidence.
DO NOT invent facts.
DO NOT invent URLs.
Every primary_url and supporting_url MUST exactly match a URL from
the supplied sources.

============================================================
EVENT DEDUPLICATION
============================================================

Multiple articles about the same underlying event must become ONE event.
Create one event with multiple supporting sources.

Do NOT re-report events that already appear in RECENT MEMORY or
RECENT REPORTS. Build on them only if something genuinely new happened.

============================================================
WHAT CHANGED
============================================================

For every selected event explain:
1. What existed before?
2. What changed?
3. Why does the change matter?

============================================================
PERSONAL RELEVANCE
============================================================

Prioritize: AI agents, agent engineering, MCP, tool calling, LLM APIs,
agent memory/state, RAG, evaluation, reliability, AI coding agents,
AI developer tools, open source AI, AI infrastructure, important AI
research, opportunities for young developers.

============================================================
SELECT 8-15 GENUINELY USEFUL EVENTS
============================================================

Do NOT artificially fill the report.
If only 5 events are genuinely important, return 5.
Include enough detail to make the report useful for actual study.

For strong events include: technical details, what changed,
implications, concrete action.

============================================================
IMPORTANCE SCALE
============================================================
10 = major ecosystem-changing development
9 = major development with significant engineering implications
8 = highly useful for serious AI engineers
7 = important and actionable
6 = useful but limited impact
5 or lower = usually exclude

============================================================
CONFIDENCE SCALE
============================================================
100 = directly confirmed by multiple strong primary sources
85-99 = strongly supported by primary evidence
70-84 = credible but incomplete corroboration
65-69 = useful but uncertain
Below 65 = exclude

============================================================
ACTION TYPES
============================================================
Use one: BUILD, TRY, LEARN, TRACK, APPLY, IGNORE
Action must be concrete. Not "Learn more about this."
Instead: "Build a small MCP server and connect one tool."

============================================================
OUTPUT FORMAT
============================================================
Return a JSON object with these fields:

{{
  "report_title": "string",
  "executive_summary": "string - 2-3 sentences",
  "events": [
    {{
      "event_id": "string - short hash or slug",
      "title": "string",
      "category": "string",
      "primary_url": "string - must match a source URL",
      "supporting_urls": ["string"],
      "importance": 1-10,
      "relevance": 1-10,
      "actionability": 1-10,
      "source_quality": 1-10,
      "confidence": 0-100,
      "tldr": "string - one dense sentence",
      "what_happened": "string",
      "what_changed": "string",
      "why_it_matters": "string",
      "key_takeaways": ["string"],
      "technical_architecture": ["string - how it works internally"],
      "technical_details": ["string"],
      "potential_impact": "string",
      "action_type": "BUILD|TRY|LEARN|TRACK|APPLY",
      "action": "string - concrete action"
    }}
  ],
  "trends": ["string"],
  "strategic_implications": ["string"],
  "build_ideas": ["string"],
  "learn_next": ["string"],
  "opportunities": ["string"],
  "things_to_ignore": ["string"]
}}
"""


def extract_json(raw: str) -> str:
    """Extract a JSON object from an LLM response.

    Handles markdown fences and prose around the payload. Raises
    ValueError if no JSON object is found.
    """
    raw = raw.strip()

    # Prefer the outermost {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Fallback: explicit ```json fences
    fence = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL
    )
    if fence:
        json.loads(fence.group(1))  # raises if still invalid
        return fence.group(1)

    raise ValueError("No valid JSON object found in LLM response")


def _as_str(value, default: str = "") -> str:
    """Coerce any LLM output to a clean string (None-safe)."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    return str(value).strip() or default


def _as_int(value, default: int, lo: int, hi: int) -> int:
    """Coerce LLM scores to a clamped int.

    Handles "9", 8.5, "91%", and None without crashing validation.
    """
    if value is None:
        return default
    try:
        return max(lo, min(hi, int(float(str(value).replace("%", "").strip()))))
    except (TypeError, ValueError):
        return default


def _as_str_list(value) -> list[str]:
    """Coerce LLM list fields to a list of clean strings.

    Non-list values (strings, nulls, dicts) become an empty list —
    malformed data must never crash the pipeline.
    """
    if not isinstance(value, list):
        return []
    return [_as_str(item) for item in value if item is not None]


def parse_report(raw_json: str) -> ResearchReport:
    """Parse Gemini's JSON response into a ResearchReport.

    Every field is coerced and clamped: LLMs sometimes emit scores as
    strings ("9"), floats (8.5), or nulls, and malformed fields must
    never crash validation downstream.
    """
    data = json.loads(extract_json(raw_json))
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON is not an object")

    events = []
    for e in data.get("events", []) or []:
        if not isinstance(e, dict):
            continue

        evidence = [
            Evidence(
                url=_as_str(ev.get("url")),
                title=_as_str(ev.get("title")),
                domain=_as_str(ev.get("domain")),
                evidence=_as_str(ev.get("evidence")),
            )
            for ev in e.get("evidence", []) or []
            if isinstance(ev, dict)
        ]

        event = ResearchEvent(
            event_id=_as_str(e.get("event_id")),
            title=_as_str(e.get("title")),
            category=_as_str(e.get("category")),
            primary_url=_as_str(e.get("primary_url")),
            supporting_urls=_as_str_list(e.get("supporting_urls")),
            importance=_as_int(e.get("importance"), 5, 1, 10),
            relevance=_as_int(e.get("relevance"), 5, 1, 10),
            actionability=_as_int(e.get("actionability"), 5, 1, 10),
            source_quality=_as_int(e.get("source_quality"), 5, 1, 10),
            confidence=_as_int(e.get("confidence"), 50, 0, 100),
            tldr=_as_str(e.get("tldr")),
            what_happened=_as_str(e.get("what_happened")),
            what_changed=_as_str(e.get("what_changed")),
            why_it_matters=_as_str(e.get("why_it_matters")),
            key_takeaways=_as_str_list(e.get("key_takeaways")),
            technical_architecture=_as_str_list(e.get("technical_architecture")),
            technical_details=_as_str_list(e.get("technical_details")),
            potential_impact=_as_str(e.get("potential_impact")),
            action_type=_as_str(e.get("action_type"), "TRACK").upper(),
            action=_as_str(e.get("action")),
            evidence=evidence,
        )

        if not event.title or not event.primary_url:
            logger.warning("Skipping malformed event (missing title or primary_url)")
            continue

        events.append(event)

    return ResearchReport(
        report_title=_as_str(data.get("report_title"), "AI Intelligence Report"),
        executive_summary=_as_str(data.get("executive_summary")),
        events=events,
        trends=_as_str_list(data.get("trends")),
        strategic_implications=_as_str_list(data.get("strategic_implications")),
        build_ideas=_as_str_list(data.get("build_ideas")),
        learn_next=_as_str_list(data.get("learn_next")),
        opportunities=_as_str_list(data.get("opportunities")),
        things_to_ignore=_as_str_list(data.get("things_to_ignore")),
    )


def _gemini_retry_wait(error_str: str, attempt: int) -> float:
    """Decide how long to wait before the next Gemini attempt.

    Honors the API's own "Please retry in Xs" hints (quota errors).
    Quota windows are per minute, so tiny waits rarely help — the
    old 5s/10s backoff kept landing in the same exhausted window.
    """
    match = re.search(r"retry in ([0-9.]+)\s*s", error_str, flags=re.IGNORECASE)
    if match:
        try:
            return min(120.0, float(match.group(1)) + 2.0)
        except ValueError:
            pass
    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
        return min(90.0, 30.0 * attempt)
    # Transient (503 UNAVAILABLE etc.): moderate backoff
    return min(60.0, 10.0 * attempt)


def analyze_with_gemini(
    candidates: list[dict],
    api_key: str,
    state: dict | None = None,
    model: str = "gemini-3.6-flash",
) -> ResearchReport:
    """Send research candidates to Gemini for analysis.

    Returns a structured ResearchReport.
    """
    client = genai.Client(api_key=api_key)

    research_data = build_research_input(candidates, state)
    prompt = build_gemini_prompt(research_data)
    logger.info(
        f"Gemini prompt size: ~{len(prompt) // 1000}K chars "
        f"(budget {GEMINI_PROMPT_CHAR_BUDGET // 1000}K)"
    )

    models = [model] + [m for m in GEMINI_MODEL_FALLBACKS if m != model]
    last_error: Exception | None = None

    for current_model in models:
        for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
            try:
                logger.info(
                    f"Gemini analysis attempt {attempt}/{GEMINI_MAX_ATTEMPTS} "
                    f"(model: {current_model})..."
                )

                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )

                report = parse_report(response.text)

                logger.info(
                    f"Gemini selected {len(report.events)} events "
                    f"(model: {current_model})"
                )

                return report

            except Exception as e:
                last_error = e
                wait = _gemini_retry_wait(str(e), attempt)
                logger.warning(
                    f"Gemini error (attempt {attempt}/{GEMINI_MAX_ATTEMPTS}, "
                    f"model: {current_model}): {e} — waiting {wait:.0f}s"
                )
                if attempt < GEMINI_MAX_ATTEMPTS:
                    time.sleep(wait)

        logger.error(
            f"Model {current_model} failed after {GEMINI_MAX_ATTEMPTS} "
            f"attempts — trying next model"
        )

    raise RuntimeError(
        f"Gemini analysis failed on all models ({', '.join(models)}): "
        f"{last_error}"
    )
