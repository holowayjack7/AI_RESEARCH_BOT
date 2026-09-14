import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

STATE_FILE = os.getenv("STATE_FILE", "data/state.json")

MAX_RESULTS_PER_QUERY = 8
MAX_CANDIDATES_FOR_RESEARCH = 24
MAX_ARTICLE_CHARS = 7000

MIN_IMPORTANCE = 6
MIN_RELEVANCE = 6
MIN_ACTIONABILITY = 5
MIN_CONFIDENCE = 65

TELEGRAM_CHUNK_SIZE = 3800

REQUEST_TIMEOUT = 20

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ============================================================
# SEARCH STRATEGY
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
}


EXCLUDED_DOMAINS = {
    "ainotdie.com",
    "claudelog.com",
    "lumichats.com",
    "stocktwits.com",
    "labmanager.com",
}


# ============================================================
# PERSONAL RESEARCH PROFILE
# ============================================================

USER_PROFILE = """
The reader is a 16-year-old developer in Georgia building toward
independent AI / Agentic Engineering.

Current roadmap:
Python
→ APIs / HTTP / JSON
→ SQL / Databases
→ LLM APIs
→ Tool Calling / MCP
→ Agent Loops
→ State
→ Graphs
→ RAG
→ Evaluation
→ Production
→ Self-improvement
→ Real Business Agents

Strong interests:
- AI agents
- agent architecture
- agent loops
- state and memory
- MCP
- tool calling
- LLM APIs
- RAG
- evaluation
- reliability
- AI coding tools
- open-source AI
- AI engineering
- practical projects
- automation
- AI business opportunities
- free/low-cost tools
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
- stock prices,
- corporate financial news,
- generic AI hype,
- celebrity/CEO news,
- generic image/video generation,
- consumer AI features with no engineering value,
- minor product updates,
- repetitive announcements.
"""


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Evidence(BaseModel):
    url: str
    title: str
    domain: str
    evidence: str


class ResearchEvent(BaseModel):
    event_id: str

    title: str
    category: str

    primary_url: str
    supporting_urls: list[str] = Field(default_factory=list)

    importance: int = Field(ge=1, le=10)
    relevance: int = Field(ge=1, le=10)
    actionability: int = Field(ge=1, le=10)
    source_quality: int = Field(ge=1, le=10)
    confidence: int = Field(ge=0, le=100)

    what_happened: str
    what_changed: str
    why_it_matters: str

    technical_details: list[str] = Field(default_factory=list)

    action_type: str
    action: str

    evidence: list[Evidence] = Field(default_factory=list)


class ResearchReport(BaseModel):
    report_title: str

    executive_summary: str

    events: list[ResearchEvent]

    trends: list[str] = Field(default_factory=list)

    strategic_implications: list[str] = Field(default_factory=list)

    build_ideas: list[str] = Field(default_factory=list)

    learn_next: list[str] = Field(default_factory=list)

    opportunities: list[str] = Field(default_factory=list)

    things_to_ignore: list[str] = Field(default_factory=list)


# ============================================================
# STATE
# ============================================================

def default_state():
    return {
        "seen_urls": [],
        "seen_event_ids": [],
        "events": [],
        "topics": {},
        "reports": [],
        "learning_queue": [],
        "last_run": None,
    }


def load_state():
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)

    if not os.path.exists(STATE_FILE):
        return default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        base = default_state()
        base.update(state)

        return base

    except Exception as e:
        print(f"State load warning: {e}")
        return default_state()


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)

    temp_file = STATE_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    os.replace(temp_file, STATE_FILE)


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    missing = []

    required = {
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }

    for name, value in required.items():
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )


# ============================================================
# URL / SOURCE UTILITIES
# ============================================================

def normalize_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()

    parsed = urlparse(url)

    if not parsed.scheme:
        return ""

    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    return clean.rstrip("/")


def domain_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        return domain
    except Exception:
        return ""


def is_trusted_domain(url: str) -> bool:
    domain = domain_from_url(url)

    if not domain:
        return False

    if domain in EXCLUDED_DOMAINS:
        return False

    return any(
        domain == trusted or domain.endswith("." + trusted)
        for trusted in TRUSTED_DOMAINS
    )


def source_quality(url: str) -> int:
    domain = domain_from_url(url)

    if domain in {
        "openai.com",
        "anthropic.com",
        "claude.com",
        "ai.google.dev",
        "deepmind.google",
        "research.google",
        "developers.googleblog.com",
        "microsoft.com",
        "azure.microsoft.com",
        "aws.amazon.com",
        "x.ai",
        "mistral.ai",
        "deepseek.com",
        "nvidia.com",
        "meta.com",
        "ai.meta.com",
    }:
        return 10

    if domain in {
        "github.com",
        "huggingface.co",
        "arxiv.org",
        "kaggle.com",
        "mlh.io",
        "hackclub.com",
    }:
        return 8

    if domain in {
        "developers.google.com",
        "cloud.google.com",
    }:
        return 9

    return 5


def make_event_id(title: str, urls: list[str]) -> str:
    normalized = "|".join(
        sorted(normalize_url(url) for url in urls if url)
    )

    raw = f"{title.lower().strip()}|{normalized}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# TAVILY
# ============================================================

def tavily_search(query: str):
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "topic": "news",
        "time_range": "week",
        "max_results": MAX_RESULTS_PER_QUERY,
        "include_domains": list(TRUSTED_DOMAINS),
        "exclude_domains": list(EXCLUDED_DOMAINS),
        "include_answer": False,
        "include_raw_content": False,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            print(f"TAVILY STATUS: {response.status_code}")

            response.raise_for_status()

            data = response.json()

            return data.get("results", [])

        except Exception as e:
            print(
                f"Tavily error "
                f"(attempt {attempt + 1}/3): {e}"
            )

            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    return []


def collect_search_results():
    print("\nSearching Tavily...\n")

    all_results = []

    for index, query in enumerate(
        SEARCH_QUERIES,
        start=1,
    ):
        print(f"Search {index}/{len(SEARCH_QUERIES)}")
        print(f"Query: {query}")

        results = tavily_search(query)

        print(f"Results: {len(results)}\n")

        for result in results:
            url = normalize_url(
                result.get("url", "")
            )

            if not url:
                continue

            if not is_trusted_domain(url):
                continue

            all_results.append({
                "title": result.get("title", "").strip(),
                "url": url,
                "content": result.get("content", "").strip(),
                "score": result.get("score", 0),
                "domain": domain_from_url(url),
                "source_quality": source_quality(url),
            })

    return all_results


# ============================================================
# SEARCH DEDUPLICATION
# ============================================================

def deduplicate_search_results(results):
    unique = {}

    for result in results:
        url = result["url"]

        if url not in unique:
            unique[url] = result
        else:
            existing = unique[url]

            if result["score"] > existing["score"]:
                unique[url] = result

    return list(unique.values())


# ============================================================
# ARTICLE FETCHING
# ============================================================

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
    ]):
        element.decompose()

    paragraphs = []

    for tag in soup.find_all([
        "article",
        "main",
        "p",
        "h1",
        "h2",
        "h3",
        "li",
    ]):
        text = tag.get_text(
            " ",
            strip=True,
        )

        if len(text) >= 40:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def fetch_article(url: str) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; AIResearchBot/3.0)"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "text/html" not in content_type:
            return None

        text = extract_text_from_html(
            response.text
        )

        if len(text) < 300:
            return None

        return text[:MAX_ARTICLE_CHARS]

    except Exception as e:
        print(
            f"Fetch failed: "
            f"{domain_from_url(url)} "
            f"{e}"
        )

        return None


# ============================================================
# CANDIDATE PREPARATION
# ============================================================

def prepare_candidates(results, state):
    seen_urls = set(state.get("seen_urls", []))

    candidates = []

    for result in results:

        if result["url"] in seen_urls:
            continue

        if result["source_quality"] < 7:
            continue

        candidates.append(result)

    candidates.sort(
        key=lambda x: (
            x["source_quality"],
            x["score"],
        ),
        reverse=True,
    )

    candidates = candidates[
        :MAX_CANDIDATES_FOR_RESEARCH
    ]

    print(
        f"\nResearch candidates: "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{index}. "
            f"{candidate['title'][:90]} "
            f"[{candidate['domain']}]"
        )

    return candidates


# ============================================================
# GEMINI
# ============================================================

def create_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


def build_research_input(candidates, state):
    sources = []

    for candidate in candidates:
        article_text = fetch_article(
            candidate["url"]
        )

        if not article_text:
            article_text = candidate["content"]

        sources.append({
            "title": candidate["title"],
            "url": candidate["url"],
            "domain": candidate["domain"],
            "source_quality": candidate["source_quality"],
            "search_score": candidate["score"],
            "content": article_text[:MAX_ARTICLE_CHARS],
        })

    historical_events = state.get(
        "events",
        []
    )[-40:]

    historical_reports = state.get(
        "reports",
        []
    )[-7:]

    learning_queue = state.get(
        "learning_queue",
        []
    )[-20:]

    return {
        "date": TODAY,
        "profile": USER_PROFILE,
        "sources": sources,
        "historical_events": historical_events,
        "historical_reports": historical_reports,
        "learning_queue": learning_queue,
    }


def build_gemini_prompt(research_data):
    return f"""
You are the intelligence analyst for a personal AI Engineering research system.

DATE:
{research_data["date"]}

USER PROFILE:
{research_data["profile"]}

INPUT SOURCES:
{json.dumps(
    research_data["sources"],
    ensure_ascii=False,
    indent=2
)}

RECENT MEMORY:
{json.dumps(
    research_data["historical_events"],
    ensure_ascii=False,
    indent=2
)}

RECENT REPORTS:
{json.dumps(
    research_data["historical_reports"],
    ensure_ascii=False,
    indent=2
)}

CURRENT LEARNING QUEUE:
{json.dumps(
    research_data["learning_queue"],
    ensure_ascii=False,
    indent=2
)}

============================================================
MISSION
============================================================

Produce a DEEP DAILY AI INTELLIGENCE REPORT.

This is NOT a news summary.

The objective is to identify information that can materially improve
the reader's AI engineering knowledge, projects, decisions, or
opportunities.

Use the supplied source content as evidence.

DO NOT invent facts.

DO NOT invent URLs.

Every primary_url and supporting_url MUST exactly match a URL from
the supplied sources.

============================================================
EVENT DEDUPLICATION
============================================================

Multiple articles about the same underlying event must become ONE event.

For example:

- product announcement
- GitHub repository
- developer documentation
- secondary article

may all describe one event.

Do not create four events.

Create one event with multiple supporting sources.

============================================================
SOURCE QUALITY
============================================================

Prefer:

1. official announcement
2. official documentation
3. official GitHub repository
4. original research paper
5. credible developer documentation

A GitHub repository alone is not proof that something is important.

Do not treat a tool directory, listicle, promotional page, or generic
AI blog as strong evidence.

============================================================
WHAT CHANGED
============================================================

For every selected event explain:

1. What existed before?
2. What changed?
3. Why does the change matter?

If the supplied evidence does not establish the previous state,
say that the previous state could not be confidently established.

============================================================
PERSONAL RELEVANCE
============================================================

Prioritize:

- AI agents
- agent engineering
- MCP
- tool calling
- LLM APIs
- agent memory/state
- RAG
- evaluation
- reliability
- AI coding agents
- AI developer tools
- open source AI
- AI infrastructure
- important AI research
- opportunities for young developers

============================================================
DEPTH
============================================================

Select approximately 8–15 genuinely useful events when evidence
supports that many.

Do NOT artificially fill the report.

If only 5 events are genuinely important, return 5.

However, do not make the report tiny just because some sources are
minor.

Include enough detail to make the report useful for actual study.

For strong events include:

- technical details
- what changed
- implications
- concrete action

============================================================
IMPORTANCE
============================================================

10 = major ecosystem-changing development

9 = major development with significant engineering implications

8 = highly useful development for serious AI engineers

7 = important and actionable

6 = useful but limited impact

5 or lower = usually exclude

============================================================
CONFIDENCE
============================================================

100 = directly confirmed by multiple strong primary sources

85–99 = strongly supported by primary evidence

70–84 = credible but incomplete corroboration

65–69 = useful but uncertain

Below 65 = exclude

============================================================
ACTION TYPES
============================================================

Use one:

BUILD
TRY
LEARN
TRACK
APPLY
IGNORE

Action must be concrete.

Bad:
"Learn more about this."

Good:
"Build a small MCP server and connect one tool to your existing
research pipeline."

============================================================
TRENDS
============================================================

Do not describe one event as a trend.

Use repeated evidence across multiple events.

Identify:

- technology trends
- developer workflow trends
- agent architecture trends
- ecosystem trends
- opportunity trends

============================================================
STRATEGIC IMPLICATIONS
============================================================

Explain what these developments mean for the reader's roadmap.

Examples:

- move MCP earlier in roadmap
- learn agent evaluation
- test a new API
- ignore a hype cycle
- modify current project architecture

============================================================
BUILD IDEAS
============================================================

Generate concrete project ideas only when strongly connected to
today's research.

Prefer ideas that teach engineering.

============================================================
LEARN NEXT
============================================================

Create a prioritized learning queue.

Avoid generic topics.

Instead of:
"Learn AI agents"

write:
"Implement a tool-calling loop with retry + state persistence."

============================================================
OPPORTUNITIES
============================================================

Include competitions, grants, programs, hackathons, open-source
opportunities, internships, fellowships or similar opportunities
only when they are genuinely relevant and eligibility is clear.

Never assume the reader qualifies.

============================================================
THINGS TO IGNORE
============================================================

Explicitly identify noisy or low-value themes from today's sources
that the reader should NOT spend time on.

============================================================
LANGUAGE
============================================================

Write the report in natural, grammatically correct Georgian.

Keep technical terms in English where appropriate:

AI agent
MCP
API
RAG
LLM
framework
repository
benchmark
tool calling
open source
SDK
GitHub
etc.

Do not translate technical terminology awkwardly.

The output must be professional, analytical and information-dense.
"""


def analyze_with_gemini(research_data):
    client = create_gemini_client()

    prompt = build_gemini_prompt(
        research_data
    )

    for attempt in range(3):

        try:
            print(
                f"\nGemini analysis "
                f"attempt {attempt + 1}/3..."
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=ResearchReport,
                ),
            )

            report = ResearchReport.model_validate_json(
                response.text
            )

            print(
                f"Gemini selected "
                f"{len(report.events)} events."
            )

            return report

        except Exception as e:

            print(
                f"Gemini error "
                f"(attempt {attempt + 1}/3): {e}"
            )

            if attempt < 2:
                time.sleep(
                    5 * (attempt + 1)
                )

    raise RuntimeError(
        "Gemini analysis failed after 3 attempts."
    )


# ============================================================
# POST-ANALYSIS VALIDATION
# ============================================================

def validate_report(report, candidates, state):
    allowed_urls = {
        candidate["url"]
        for candidate in candidates
    }

    valid_events = []

    seen_event_ids = set(
        state.get("seen_event_ids", [])
    )

    for event in report.events:

        if event.primary_url not in allowed_urls:
            print(
                f"Rejected event: invalid "
                f"primary URL -> "
                f"{event.primary_url}"
            )
            continue

        supporting = []

        for url in event.supporting_urls:
            if url in allowed_urls:
                supporting.append(url)

        event.supporting_urls = supporting

        event.event_id = (
            event.event_id.strip()
            or make_event_id(
                event.title,
                [event.primary_url]
                + event.supporting_urls,
            )
        )

        if event.event_id in seen_event_ids:
            continue

        if event.importance < MIN_IMPORTANCE:
            continue

        if event.relevance < MIN_RELEVANCE:
            continue

        if event.actionability < MIN_ACTIONABILITY:
            continue

        if event.confidence < MIN_CONFIDENCE:
            continue

        python_quality = source_quality(
            event.primary_url
        )

        event.source_quality = min(
            event.source_quality,
            python_quality,
        )

        if event.source_quality < 7:
            continue

        valid_events.append(event)

    valid_events.sort(
        key=lambda e: (
            e.importance,
            e.relevance,
            e.actionability,
            e.confidence,
        ),
        reverse=True,
    )

    report.events = valid_events[:15]

    return report


# ============================================================
# TELEGRAM HTML
# ============================================================

def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_event(index, event):
    lines = []

    lines.append(
        f"<b>{index}. "
        f"{escape_html(event.title)}</b>"
    )

    lines.append(
        f"🏷 <b>{escape_html(event.category)}</b>"
    )

    lines.append(
        f"Importance: <b>{event.importance}/10</b> | "
        f"Relevance: <b>{event.relevance}/10</b> | "
        f"Actionability: <b>{event.actionability}/10</b>"
    )

    lines.append(
        f"Confidence: <b>{event.confidence}%</b> | "
        f"Source quality: <b>{event.source_quality}/10</b>"
    )

    lines.append("")

    lines.append(
        "<b>რა მოხდა</b>\n"
        + escape_html(event.what_happened)
    )

    lines.append("")

    lines.append(
        "<b>რა შეიცვალა</b>\n"
        + escape_html(event.what_changed)
    )

    lines.append("")

    lines.append(
        "<b>რატომ არის მნიშვნელოვანი</b>\n"
        + escape_html(event.why_it_matters)
    )

    if event.technical_details:
        lines.append("")
        lines.append("<b>Technical details</b>")

        for detail in event.technical_details:
            lines.append(
                "• " + escape_html(detail)
            )

    lines.append("")

    lines.append(
        f"<b>Action: "
        f"{escape_html(event.action_type)}</b>\n"
        f"{escape_html(event.action)}"
    )

    lines.append("")

    lines.append(
        f"🔗 <a href=\"{event.primary_url}\">"
        f"Primary source</a>"
    )

    for url in event.supporting_urls[:4]:
        lines.append(
            f"• <a href=\"{url}\">"
            f"Supporting source</a>"
        )

    return "\n".join(lines)


def build_telegram_report(report):
    lines = []

    lines.append(
        f"<b>AI INTELLIGENCE REPORT</b>\n"
        f"{TODAY}"
    )

    lines.append("")

    lines.append(
        "<b>Executive Summary</b>"
    )

    lines.append(
        escape_html(
            report.executive_summary
        )
    )

    lines.append("")

    lines.append(
        "<b>🔥 TOP INTELLIGENCE</b>"
    )

    for index, event in enumerate(
        report.events,
        start=1,
    ):
        lines.append(
            format_event(index, event)
        )

        lines.append(
            "\n━━━━━━━━━━━━━━━━━━\n"
        )

    if report.trends:
        lines.append(
            "<b>📈 TRENDS</b>"
        )

        for trend in report.trends:
            lines.append(
                "• " + escape_html(trend)
            )

        lines.append("")

    if report.strategic_implications:
        lines.append(
            "<b>🧠 STRATEGIC IMPLICATIONS</b>"
        )

        for item in report.strategic_implications:
            lines.append(
                "• " + escape_html(item)
            )

        lines.append("")

    if report.build_ideas:
        lines.append(
            "<b>🛠 BUILD IDEAS</b>"
        )

        for item in report.build_ideas:
            lines.append(
                "• " + escape_html(item)
            )

        lines.append("")

    if report.learn_next:
        lines.append(
            "<b>📚 LEARN NEXT</b>"
        )

        for index, item in enumerate(
            report.learn_next,
            start=1,
        ):
            lines.append(
                f"{index}. "
                + escape_html(item)
            )

        lines.append("")

    if report.opportunities:
        lines.append(
            "<b>🎯 OPPORTUNITIES</b>"
        )

        for item in report.opportunities:
            lines.append(
                "• " + escape_html(item)
            )

        lines.append("")

    if report.things_to_ignore:
        lines.append(
            "<b>🚫 IGNORE / LOW VALUE</b>"
        )

        for item in report.things_to_ignore:
            lines.append(
                "• " + escape_html(item)
            )

    lines.append("")
    lines.append(
        "<i>AI Research Bot — Daily Intelligence</i>"
    )

    return "\n".join(lines)


# ============================================================
# TELEGRAM CHUNKING
# ============================================================

def split_message(text, max_length=TELEGRAM_CHUNK_SIZE):
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    blocks = text.split("\n━━━━━━━━━━━━━━━━━━\n")

    for block in blocks:

        separator = (
            "\n━━━━━━━━━━━━━━━━━━\n"
        )

        if len(current) + len(block) + len(separator) <= max_length:
            if current:
                current += separator

            current += block

        else:
            if current:
                chunks.append(current)

            if len(block) <= max_length:
                current = block
            else:
                while len(block) > max_length:
                    chunks.append(
                        block[:max_length]
                    )
                    block = block[max_length:]

                current = block

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram(text):
    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    chunks = split_message(text)

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        for attempt in range(3):

            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                print(
                    f"TELEGRAM STATUS: "
                    f"{response.status_code}"
                )

                response.raise_for_status()

                print(
                    f"Telegram message "
                    f"{index}/{len(chunks)} sent."
                )

                break

            except Exception as e:

                print(
                    f"Telegram error "
                    f"(attempt {attempt + 1}/3): "
                    f"{e}"
                )

                if attempt == 2:
                    raise

                time.sleep(
                    3 * (attempt + 1)
                )


# ============================================================
# STATE UPDATE
# ============================================================

def update_state(
    state,
    report,
    candidates,
):
    sent_urls = [
        candidate["url"]
        for candidate in candidates
    ]

    state["seen_urls"].extend(
        sent_urls
    )

    state["seen_urls"] = list(
        dict.fromkeys(
            state["seen_urls"]
        )
    )[-2000:]

    for event in report.events:

        state["seen_event_ids"].append(
            event.event_id
        )

        state["events"].append(
            event.model_dump()
        )

        topic = event.category

        state["topics"][topic] = (
            state["topics"].get(topic, 0)
            + 1
        )

    state["seen_event_ids"] = list(
        dict.fromkeys(
            state["seen_event_ids"]
        )
    )[-1000:]

    state["events"] = state["events"][-100:]

    state["reports"].append({
        "date": TODAY,
        "summary": report.executive_summary,
        "events": [
            event.model_dump()
            for event in report.events
        ],
        "trends": report.trends,
    })

    state["reports"] = state[
        "reports"
    ][-14:]

    state["learning_queue"] = (
        report.learn_next[-20:]
    )

    state["last_run"] = (
        datetime.now(timezone.utc)
        .isoformat()
    )


# ============================================================
# MAIN RESEARCH CYCLE
# ============================================================

def run():
    print(
        "\n"
        "========================================\n"
        "      AI RESEARCH BOT v3\n"
        "========================================\n"
    )

    validate_config()

    print("Configuration OK")
    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    state = load_state()

    # --------------------------------------------------------
    # 1. SEARCH
    # --------------------------------------------------------

    raw_results = collect_search_results()

    print(
        f"\nTotal raw results: "
        f"{len(raw_results)}"
    )

    # --------------------------------------------------------
    # 2. URL DEDUP
    # --------------------------------------------------------

    unique_results = (
        deduplicate_search_results(
            raw_results
        )
    )

    print(
        f"After URL dedup: "
        f"{len(unique_results)}"
    )

    # --------------------------------------------------------
    # 3. FILTER + PERSONAL MEMORY
    # --------------------------------------------------------

    candidates = prepare_candidates(
        unique_results,
        state,
    )

    if not candidates:
        print(
            "\nNo new research candidates."
        )

        state["last_run"] = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        save_state(state)

        return

    # --------------------------------------------------------
    # 4. FETCH + DEEP ANALYSIS
    # --------------------------------------------------------

    research_data = build_research_input(
        candidates,
        state,
    )

    report = analyze_with_gemini(
        research_data
    )

    # --------------------------------------------------------
    # 5. VALIDATE AI OUTPUT
    # --------------------------------------------------------

    report = validate_report(
        report,
        candidates,
        state,
    )

    print(
        f"\nValidated events: "
        f"{len(report.events)}"
    )

    # --------------------------------------------------------
    # 6. IMPORTANT:
    # If there is genuinely nothing important,
    # DO NOT SEND TELEGRAM NOISE.
    # --------------------------------------------------------

    if not report.events:

        print(
            "\nNo sufficiently important "
            "events today."
        )

        state["last_run"] = (
            datetime.now(timezone.utc)
            .isoformat()
        )

        save_state(state)

        return

    # --------------------------------------------------------
    # 7. TELEGRAM
    # --------------------------------------------------------

    telegram_message = (
        build_telegram_report(report)
    )

    send_telegram(
        telegram_message
    )

    # --------------------------------------------------------
    # 8. ONLY AFTER SUCCESSFUL TELEGRAM SEND:
    # update persistent memory
    # --------------------------------------------------------

    update_state(
        state,
        report,
        candidates,
    )

    save_state(state)

    print(
        "\nState saved."
    )

    print(
        "Research cycle completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        run()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        print(
            "\nFATAL ERROR:",
            repr(e)
        )

        raise