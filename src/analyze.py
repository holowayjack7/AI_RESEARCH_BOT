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
    GEMINI_CHAIN_PASSES,
    GEMINI_CONTENT_CHARS,
    GEMINI_INTER_PASS_SECONDS,
    GEMINI_MAX_ATTEMPTS,
    GEMINI_MODEL,
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

The reader wants BROAD AI coverage: AI agents, AI engineering,
model releases, research papers, developer tools, and major
industry news — but NEVER hype, rumors, or financial noise.

Strong interests:
- AI agents, agent architecture, agent loops
- state and memory, MCP, tool calling
- LLM APIs, RAG, evaluation, reliability
- AI coding tools, open-source AI
- AI engineering, practical projects, automation
- AI business opportunities, free/low-cost tools
- opportunities accessible to teenagers 16+

The reader does NOT want hype, rumors, or empty speculation —
but genuine AI industry news with technical substance is welcome.

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
class CandidateAnalysis:
    """Per-article critical analysis from the pre-selection pass.

    One entry per supplied source; the decision layer in pipeline.py
    uses is_relevant / should_send to gate events built from that
    article, so a story is judged on its own evidence before it can
    become an event.
    """
    title: str = ""
    url: str = ""
    is_relevant: bool = True
    source_quality: int = 5
    factuality_level: str = "single_source"
    importance_score: int = 5
    classification: list[str] = field(default_factory=list)
    verified_facts: list[str] = field(default_factory=list)
    interpretation: str = ""
    uncertainty: str = ""
    counter_argument: str = ""
    actionable_takeaway: str = ""
    should_send: bool = True


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
    # Critical-analysis fields (additive; defaults keep old state data valid)
    factuality_level: str = "single_source"
    classification: list[str] = field(default_factory=list)
    verified_facts: list[str] = field(default_factory=list)
    interpretation: str = ""
    uncertainty: str = ""
    counter_argument: str = ""


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
    # Per-article critical analyses (one per supplied source)
    candidate_analyses: list[CandidateAnalysis] = field(default_factory=list)


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

QUALITY BAR: every event must pass this test — "Would a busy senior
AI engineer forward this to a colleague?" If not, exclude it.

============================================================
CRITICAL THINKING FRAMEWORK (EVERY ARTICLE, BEFORE ANY DECISION)
============================================================

Analyze every supplied article along these dimensions:

- FACTS: what is objectively verified by the source text? Concrete
  numbers, names, versions, benchmarks, code. Keep strictly separate
  from interpretation.
- INTERPRETATION: what could this information potentially mean?
- UNCERTAINTY: what is unknown, speculative, or unsupported?
- COUNTERARGUMENT: the strongest argument against the initial
  interpretation. If you cannot produce one, think harder.
- INCENTIVES: could the source have marketing, financial, or
  engagement motives? Vendor announcements deserve extra scrutiny.
- LONGEVITY: temporary hype, durable technical development, or a
  potentially lasting trend?
- TRANSFERABILITY: does the knowledge apply beyond this specific
  product or announcement?
- EVIDENCE: how strong and independently verifiable is the evidence?

============================================================
CLASSIFICATION (EVERY EVENT, ONE OR MORE)
============================================================

Classify each selected event into one or more of:
- "Temporary Trend"
- "Real Technical Skill"
- "Real Business Opportunity"
- "Long-Term Career Value"
- "General News"

Prioritize durable knowledge and practical consequences over hype.

============================================================
FACTUALITY LEVELS (EVERY EVENT AND ARTICLE)
============================================================

factuality_level must be exactly one of:
- "verified": confirmed by multiple independent strong sources
- "corroborated": primary source plus supporting evidence
- "single_source": one credible source, unconfirmed elsewhere
- "speculative": claim or forecast without solid evidence

Label speculation honestly. Speculation is acceptable ONLY when
clearly labeled and it provides meaningful strategic insight.

============================================================
DECISION RULES (REJECT OR DEPRIORITIZE)
============================================================

Reject or deprioritize information if:
- it duplicates another article or something in RECENT MEMORY;
- it has low relevance to AI engineering, agents, automation, or
  AI careers;
- it is mostly speculation without evidence (labeled speculation
  with real strategic insight is still acceptable);
- it is a minor product update with no meaningful consequences;
- it repeats information already sent recently;
- it has no practical, technical, business, or career value.

The importance score alone NEVER decides: weigh evidence quality,
long-term relevance, practical value, uniqueness, and user relevance.

WORKFLOW: first produce one candidate_analyses entry per supplied
source (applying these rules through is_relevant and should_send),
then select events ONLY from sources whose should_send is true.

============================================================
REPORT STRUCTURE (MANDATORY — EVERY EVENT)
============================================================

Each event's fields must collectively follow this exact structure:

1) EXECUTIVE IMPACT (tldr)
   1-2 sentences: what launched or broke, and why it matters
   immediately. State the concrete capability change.

2) KEY TECHNICAL BREAKDOWN
   - technical_architecture = "Architecture and Specs": frameworks,
     model benchmarks, context limits, latency/throughput changes,
     parameter counts, pricing. Concrete numbers and names only.
     Omit entirely if the source provides none.
   - technical_details = "Implementation Logic": API usage patterns,
     pseudo-code or structural code snippets, config flags,
     integration points — as applicable to the source.

3) ACTIONABLE TAKEAWAYS
   - action: a direct instruction for applying this to real software
     workflows or projects this week.
   - potential_impact: commercial potential and strategic advantage,
     stated concretely (who gains what, by how much).

4) VERIFIED RESOURCES
   - primary_url + supporting_urls: source code repositories,
     research papers, official releases. Only URLs present in the
     supplied sources.

============================================================
CONTENT RULES (HARD BANS)
============================================================

- ZERO emojis, icons, or decorative symbols in ANY field. Plain
  technical text only.
- Banned phrases: "In recent news", "AI is evolving rapidly",
  "It is important to note", "This is significant because",
  "In a major move", "Needless to say".
- Banned adjectives: game-changing, revolutionary, cutting-edge,
  groundbreaking.
- Every field must carry concrete technical data (numbers, names,
  versions, benchmarks, code constructs). A field that would be
  generic must be made specific or shortened.

============================================================
OUTPUT LANGUAGE (MANDATORY — HYBRID GEORGIAN/ENGLISH)
============================================================

The reader is Georgian. ALL human-readable text must be written in
a hybrid language: ~70% proper, well-structured Georgian and ~30%
clear English. Maintain a professional intelligence-analyst tone
in natural, correct Georgian.

In GEORGIAN (~70% of words):
- all analysis and explanation prose: tldr, what_happened,
  what_changed, why_it_matters, potential_impact, action,
  interpretation, uncertainty, counter_argument, verified_facts,
  candidate_analyses prose fields (interpretation, uncertainty,
  counter_argument, actionable_takeaway, verified_facts)
- every list item: key_takeaways, trends, strategic_implications,
  build_ideas, learn_next, opportunities, things_to_ignore
- report_title and executive_summary
- category (short Georgian label)

In ENGLISH, kept verbatim — never transliterate, never translate
(~30% of words, the technical layer):
- engineering terminology: model, API, SDK, agent, MCP, RAG, tool
  calling, token, context window, latency, throughput, benchmark,
  fine-tuning, inference, framework, retrieval, memory
- product/model/library names, version numbers, benchmarks,
  code identifiers and snippets (inside technical_details and
  technical_architecture, English dominates naturally)
- machine-read fields EXACTLY as specified: classification values
  ("Real Technical Skill" etc.), factuality_level ("verified" etc.),
  action_type (BUILD/TRY/LEARN/TRACK/APPLY/IGNORE), event_id
- URLs, domain names

Do not mix scripts inside a single technical term. Georgian text
must not be a word-for-word translation — write it as a Georgian
engineer would naturally explain it to a colleague.

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
SELECT 5-10 HIGH-SIGNAL EVENTS (DENSITY OVER VOLUME)
============================================================

Do NOT artificially fill the report. If only 3 events pass the
quality bar, return 3. Skip: incremental minor releases, marketing
renames, version bumps without capability changes, and anything
already covered in RECENT MEMORY.

Prefer: official releases and docs, papers with measurable results,
tools with immediately actionable capabilities, architecture shifts.

Every event must be information-dense: concrete numbers, named
technologies, specific capabilities. If a field would be generic,
make it specific or shorten it.

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
CONCISENESS DOCTRINE (HARD LIMITS)
============================================================

Respect these word limits exactly — conciseness is a feature:
- executive_summary: max 3 short sentences (~60 words)
- tldr: 1-2 sentences, max 30 words — what launched or broke, and
  why it matters immediately (Executive Impact)
- what_happened / what_changed / why_it_matters / potential_impact:
  max 30 words each, past/present/future tense respectively
- key_takeaways: 2-4 items, max 15 words each, each a concrete fact
- technical_architecture: 0-3 items, max 15 words each (omit if the
  source gives no real architectural information)
- technical_details: 2-4 items, max 12 words each (numbers, names)
- action: ONE imperative sentence, max 25 words, doable this week

Banned filler phrases: "In recent news", "AI is evolving rapidly",
"It is important to note", "This is significant because",
"In a major move". No repeating the title, no vague praise,
no restating the obvious, no emojis or decorative symbols anywhere.

============================================================
OUTPUT FORMAT
============================================================
Return a JSON object with these fields:

{{
  "report_title": "string",
  "executive_summary": "string - max 3 short sentences, no preamble",
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
      "factuality_level": "verified|corroborated|single_source|speculative",
      "classification": ["one or more of: Temporary Trend, Real Technical Skill, Real Business Opportunity, Long-Term Career Value, General News"],
      "verified_facts": ["2-5 items, max 15 words each: only objectively verified claims"],
      "interpretation": "string - max 30 words, clearly separated from facts",
      "uncertainty": "string - max 25 words: what is unknown, speculative, or unsupported",
      "counter_argument": "string - max 25 words: strongest argument against your interpretation",
      "tldr": "string - 1-2 sentences, max 30 words: what launched/broke and why it matters immediately",
      "what_happened": "string - max 30 words, concrete facts",
      "what_changed": "string - max 25 words, before -> after",
      "why_it_matters": "string - max 30 words, for THIS reader",
      "key_takeaways": ["string - 2-4 items, max 15 words each"],
      "technical_architecture": ["string - 0-3 items, max 15 words each"],
      "technical_details": ["string - 2-4 items, max 12 words each"],
      "potential_impact": "string - max 25 words",
      "action_type": "BUILD|TRY|LEARN|TRACK|APPLY",
      "action": "string - ONE imperative sentence, max 25 words"
    }}
  ],
  "trends": ["string"],
  "strategic_implications": ["string"],
  "build_ideas": ["string"],
  "learn_next": ["string"],
  "opportunities": ["string"],
  "things_to_ignore": ["string"],
  "candidate_analyses": [
    {{
      "title": "string - copied from the source",
      "url": "string - EXACTLY matches a supplied source URL",
      "is_relevant": true,
      "source_quality": 1-10,
      "factuality_level": "verified|corroborated|single_source|speculative",
      "importance_score": 1-10,
      "classification": ["one or more categories from the taxonomy"],
      "verified_facts": ["only objectively verified claims"],
      "interpretation": "max 25 words",
      "uncertainty": "max 25 words",
      "counter_argument": "max 25 words",
      "actionable_takeaway": "max 25 words",
      "should_send": true
    }}
  ]
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


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
    "|[\u2190-\u21FF\u2B00-\u2BFF\u25A0-\u25FF\u2700-\u27BF]"
)


def _strip_emoji(text: str) -> str:
    """Remove emojis/icons/decorative symbols from LLM output.

    The report contract bans them outright; this guarantees the
    delivered text complies even if the model slips one in.
    """
    cleaned = _EMOJI_RE.sub("", text or "")
    return re.sub(r"  +", " ", cleaned).strip()


def _as_str(value, default: str = "") -> str:
    """Coerce any LLM output to a clean, emoji-free string (None-safe)."""
    if value is None:
        return default
    if isinstance(value, str):
        return _strip_emoji(value) or default
    return _strip_emoji(str(value)) or default


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


def _as_bool(value, default: bool) -> bool:
    """Coerce LLM booleans ("true", True, 1, None-safe)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


FACTUALITY_LEVELS = {"verified", "corroborated", "single_source", "speculative"}


def _as_factuality(value, default: str = "single_source") -> str:
    """Coerce factuality_level to one of the canonical levels.

    Free-text variants ("Verified", "highly speculative") are mapped
    to the closest canonical level instead of crashing the gate.
    """
    raw = _as_str(value, default).lower().strip()
    if raw in FACTUALITY_LEVELS:
        return raw
    if "verif" in raw:
        return "verified"
    if "corrob" in raw:
        return "corroborated"
    if "single" in raw:
        return "single_source"
    if "specul" in raw:
        return "speculative"
    return default if default in FACTUALITY_LEVELS else "single_source"


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
            factuality_level=_as_factuality(e.get("factuality_level")),
            classification=_as_str_list(e.get("classification")),
            verified_facts=_as_str_list(e.get("verified_facts")),
            interpretation=_as_str(e.get("interpretation")),
            uncertainty=_as_str(e.get("uncertainty")),
            counter_argument=_as_str(e.get("counter_argument")),
        )

        if not event.title or not event.primary_url:
            logger.warning("Skipping malformed event (missing title or primary_url)")
            continue

        events.append(event)

    candidate_analyses = []
    for a in data.get("candidate_analyses", []) or []:
        if not isinstance(a, dict):
            continue
        analysis = CandidateAnalysis(
            title=_as_str(a.get("title")),
            url=_as_str(a.get("url")),
            is_relevant=_as_bool(a.get("is_relevant"), True),
            source_quality=_as_int(a.get("source_quality"), 5, 1, 10),
            factuality_level=_as_factuality(a.get("factuality_level")),
            importance_score=_as_int(a.get("importance_score"), 5, 1, 10),
            classification=_as_str_list(a.get("classification")),
            verified_facts=_as_str_list(a.get("verified_facts")),
            interpretation=_as_str(a.get("interpretation")),
            uncertainty=_as_str(a.get("uncertainty")),
            counter_argument=_as_str(a.get("counter_argument")),
            actionable_takeaway=_as_str(a.get("actionable_takeaway")),
            should_send=_as_bool(a.get("should_send"), True),
        )
        if not analysis.url:
            logger.warning("Skipping malformed candidate analysis (missing url)")
            continue
        candidate_analyses.append(analysis)

    return enforce_conciseness(ResearchReport(
        report_title=_as_str(data.get("report_title"), "AI Intelligence Report"),
        executive_summary=_as_str(data.get("executive_summary")),
        events=events,
        trends=_as_str_list(data.get("trends")),
        strategic_implications=_as_str_list(data.get("strategic_implications")),
        build_ideas=_as_str_list(data.get("build_ideas")),
        learn_next=_as_str_list(data.get("learn_next")),
        opportunities=_as_str_list(data.get("opportunities")),
        things_to_ignore=_as_str_list(data.get("things_to_ignore")),
        candidate_analyses=candidate_analyses,
    ))


def _is_permanent_model_error(error_str: str) -> bool:
    """True for errors that retrying the same model can never fix.

    Retired/deprecated models return 404 NOT_FOUND forever (e.g.
    "no longer available to new users"), blocked models return
    permission errors, an invalid API key returns 400
    API_KEY_INVALID, and a DAILY quota exhaustion (quotaId
    *PerDay*) stays exhausted until Google's daily reset — waiting
    seconds cannot help. In all these cases the correct move is to
    fall to the next model in the chain immediately: fallback models
    carry their own separate daily quota buckets.
    """
    s = error_str.lower()
    return (
        "not_found" in s
        or "404" in s
        or "no longer available" in s
        or "does not exist" in s
        or "is not supported" in s
        or "permission denied" in s
        or "permission_denied" in s
        or "api key not valid" in s
        or "api_key_invalid" in s
        or "unauthenticated" in s
        or "perday" in s
        or "per_day" in s
    )


def _gemini_retry_wait(error_str: str, attempt: int) -> float:
    """Exponential backoff before the next Gemini attempt.

    Honors the API's own "Please retry in Xs" hints (quota errors).
    Quota windows are per minute, so a fixed tiny backoff kept landing
    in the same exhausted window; waits now grow exponentially from
    GEMINI_RETRY_BASE_SECONDS (capped) and respect server hints.
    """
    match = re.search(r"retry in ([0-9.]+)\s*s", error_str, flags=re.IGNORECASE)
    if match:
        try:
            return min(120.0, float(match.group(1)) + 2.0)
        except ValueError:
            pass
    from config import GEMINI_RETRY_BASE_SECONDS

    base = max(1.0, float(GEMINI_RETRY_BASE_SECONDS))
    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
        # Quota: grow base -> 2x -> 4x ... so later attempts can land
        # in a fresh per-minute window
        return min(120.0, base * (2 ** max(0, attempt - 1)))
    # Transient (503 UNAVAILABLE etc.): exponential as well
    return min(60.0, base * attempt)


# ============================================================
# CONCISENESS ENFORCEMENT (defensive post-processing)
# ============================================================

def _clip_words(text: str, max_words: int) -> str:
    """Clip text to max_words at a word boundary."""
    text = (text or "").strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _clip_items(items: list[str], max_count: int, max_words: int) -> list[str]:
    """Cap list length and per-item words, dropping empties."""
    return [
        _clip_words(item, max_words)
        for item in (items or [])[:max_count]
        if item and item.strip()
    ]


def enforce_conciseness(report: ResearchReport) -> ResearchReport:
    """Clip any fields that exceed the conciseness doctrine.

    The prompt demands tight word limits; this guarantees the
    delivered report stays compact even when the model ignores them.
    """
    report.executive_summary = _clip_words(report.executive_summary, 60)

    for event in report.events:
        event.tldr = _clip_words(event.tldr, 30)
        event.what_happened = _clip_words(event.what_happened, 30)
        event.what_changed = _clip_words(event.what_changed, 25)
        event.why_it_matters = _clip_words(event.why_it_matters, 30)
        event.potential_impact = _clip_words(event.potential_impact, 25)
        event.action = _clip_words(event.action, 25)
        event.key_takeaways = _clip_items(event.key_takeaways, 4, 15)
        event.technical_architecture = _clip_items(
            event.technical_architecture, 3, 15
        )
        event.technical_details = _clip_items(event.technical_details, 4, 12)
        event.verified_facts = _clip_items(event.verified_facts, 5, 15)
        event.classification = _clip_items(event.classification, 3, 10)
        event.interpretation = _clip_words(event.interpretation, 30)
        event.uncertainty = _clip_words(event.uncertainty, 25)
        event.counter_argument = _clip_words(event.counter_argument, 25)

    for analysis in report.candidate_analyses:
        analysis.verified_facts = _clip_items(analysis.verified_facts, 5, 15)
        analysis.classification = _clip_items(analysis.classification, 3, 10)
        analysis.interpretation = _clip_words(analysis.interpretation, 25)
        analysis.uncertainty = _clip_words(analysis.uncertainty, 25)
        analysis.counter_argument = _clip_words(analysis.counter_argument, 25)
        analysis.actionable_takeaway = _clip_words(
            analysis.actionable_takeaway, 25
        )

    return report


def analyze_with_gemini(
    candidates: list[dict],
    api_key: str,
    state: dict | None = None,
    model: str = GEMINI_MODEL,
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
    saw_transient_error = False

    for pass_num in range(1, GEMINI_CHAIN_PASSES + 1):
        for current_model in models:
            for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
                try:
                    logger.info(
                        f"Gemini analysis attempt {attempt}/{GEMINI_MAX_ATTEMPTS} "
                        f"(model: {current_model}, pass {pass_num}/"
                        f"{GEMINI_CHAIN_PASSES})..."
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
                    if _is_permanent_model_error(str(e)):
                        logger.warning(
                            f"Model {current_model} permanently unavailable "
                            f"(404/invalid key/permission) — falling to "
                            f"next model immediately"
                        )
                        break
                    saw_transient_error = True
                    wait = _gemini_retry_wait(str(e), attempt)
                    logger.warning(
                        f"Gemini error (attempt {attempt}/{GEMINI_MAX_ATTEMPTS}, "
                        f"model: {current_model}): {e} — waiting {wait:.0f}s"
                    )
                    if attempt < GEMINI_MAX_ATTEMPTS:
                        time.sleep(wait)

            logger.error(
                f"Model {current_model} failed — trying next model"
            )

        # More passes remain AND at least one failure was transient
        # (retrying a chain whose every model failed permanently can
        # never succeed): pause so quota windows reset and 503 capacity
        # storms can clear, then try the whole chain again.
        if pass_num < GEMINI_CHAIN_PASSES and saw_transient_error:
            logger.warning(
                f"All {len(models)} model(s) failed on pass {pass_num}/"
                f"{GEMINI_CHAIN_PASSES} — pausing "
                f"{GEMINI_INTER_PASS_SECONDS}s before retrying the chain"
            )
            time.sleep(GEMINI_INTER_PASS_SECONDS)
        else:
            break
            logger.warning(
                f"All {len(models)} model(s) failed on pass {pass_num}/"
                f"{GEMINI_CHAIN_PASSES} — pausing "
                f"{GEMINI_INTER_PASS_SECONDS}s before retrying the chain"
            )
            time.sleep(GEMINI_INTER_PASS_SECONDS)

    raise RuntimeError(
        f"Gemini analysis failed on all models ({', '.join(models)}): "
        f"{last_error}"
    )
