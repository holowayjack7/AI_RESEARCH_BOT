# AI Research Bot

An autonomous AI research & news agency agent: collects AI research papers and news, curates them, analyzes with Gemini, publishes structured reports, and delivers a daily intelligence digest to Telegram.

## What It Does

1. **Collects** — Tavily web search (news from trusted domains), arXiv API, and Hugging Face Daily Papers (the latter two need no API keys)
2. **Curates** — URL + arXiv-id cross-source deduplication, source-quality scoring, seen-item state tracking, quality thresholds
3. **Analyzes** — Gemini produces a structured intelligence report (TL;DR, key takeaways, technical architecture, potential impact, actions)
4. **Publishes** — Markdown report, JSON, newsletter digest, and short-form social posts under `data/reports/YYYY-MM-DD/`
5. **Delivers** — Telegram digest with a modern UI: expandable blockquotes, medal ranks, impact scores, in-this-issue index. Telegram is optional — without credentials the bot runs export-only
6. **Remembers** — persistent state in `data/state.json` prevents re-reporting known stories across runs; only delivered URLs are marked seen, so analyzed-but-unreported items can resurface while fresh

Pipeline: `search → collect → deduplicate → filter → fetch → analyze → validate → publish → deliver`

## Project Structure

```
ai-research-bot/
│
├── main.py              # Entry point (full run, --simulate, --check-telegram)
├── config.py            # Environment variables + feature flags
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── tests/test_review.py # Regression test suite
├── .github/workflows/   # Daily scheduled runs (tests gate delivery)
│
├── src/
│   ├── net.py           # Rate-limited HTTP with retries/backoff
│   ├── sources.py       # arXiv + Hugging Face Papers collectors
│   ├── search.py        # Tavily web search
│   ├── fetcher.py       # Article fetching + text extraction
│   ├── analyze.py       # Gemini analysis, crash-proof report parsing
│   ├── pipeline.py      # Orchestration, validation, persistent state
│   ├── publish.py       # Markdown / JSON / newsletter / social export
│   ├── deliver.py       # Telegram formatting + resilient sending
│   └── logger.py        # Console + structured JSONL logging
│
└── data/
    ├── state.json       # Persistent memory (seen URLs/ids/keys, reports)
    ├── logs/bot.jsonl   # Structured run log
    └── reports/         # Published reports (md/json/newsletter/social)
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# .venv\Scripts\activate    # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your .env file

```bash
cp .env.example .env
```

Fill in the keys — only `GEMINI_API_KEY` is strictly required:

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | ✅ | Analysis (Gemini) |
| `TAVILY_API_KEY` | optional | Adds web/news search on top of arXiv + HF |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | optional | Chat delivery |
| `GEMINI_MODEL` | optional | Defaults to `gemini-3.6-flash` |

### 4. Verify Telegram delivery (recommended once)

```bash
python main.py --check-telegram
```

Sends a test message to your chat. If it fails, the most common cause is that your bot has never received a message from you — open the bot in Telegram and press **Start** first.

### 5. Run the bot

```bash
python main.py             # full run (collect → analyze → publish → deliver)
python main.py --simulate  # offline end-to-end test (no keys, no network)
```

The workflow (`.github/workflows/research.yml`) runs the test suite before every scheduled run, so a broken build can never silently skip delivery.

## Run automatically with GitHub Actions

The included workflow runs the bot **every day at 14:00 Tbilisi time (10:00 UTC)** (plus manual triggers from the Actions tab) and auto-commits the updated state to `data/state.json`.

### 1. Set up repository secrets

In your GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Required | Value |
|--------|----------|-------|
| `GEMINI_API_KEY` | ✅ | Your Gemini API key ([get one free](https://aistudio.google.com/apikey)) |
| `TELEGRAM_BOT_TOKEN` | for delivery | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | for delivery | Your chat id (send the bot a message, then check `https://api.telegram.org/bot<TOKEN>/getUpdates`) |
| `TAVILY_API_KEY` | optional | Adds web/news search on top of arXiv + Hugging Face |
| `GEMINI_MODEL` | optional | Defaults to `gemini-3.6-flash` — a missing secret is safe |

> **Important:** open your bot in Telegram and press **Start** (or send `/start`) once. A bot can only message a chat that has contacted it first — this is the most common cause of failed delivery.

### 2. Trigger a run

- **Automatic:** daily on schedule (10:00 UTC)
- **Manual:** Actions tab → **AI Research Bot** → **Run workflow**

### 3. What a run does

1. Installs dependencies (cached)
2. Runs the test suite — failures abort the run before delivery
3. Collects → curates → analyzes → publishes report files
4. Delivers the digest to your Telegram (if credentials are set) — on quiet days a short "📭 no new intelligence" heartbeat is sent instead, so you get a message every day
5. Commits the updated `data/state.json` so the next run remembers what was already reported

Overlapping runs are queued (not killed), and each run is capped at 15 minutes.

## Telegram UI

Each report renders as compact cards:

- **Header card** — date, curated-event count, expandable executive summary, 📋 *in this issue* index
- **Event cards** — 🥇🥈🥉 medal ranks, category + source-type chips, ✅ confidence, monospace score bars with a weighted ◆ impact score, ⚡ TL;DR first, expandable background (what happened / changed / why), takeaways, architecture, details, action callout, source links
- **Report sections** — 📈 Trends, 🧭 Strategic implications, 🛠 Build ideas, 📚 Learn next, 🎯 Opportunities, 🚫 Things to ignore

Delivery is resilient: per-chunk retries honoring Telegram's `retry_after`, plain-text fallback on HTML parse errors, and a failing chunk never aborts the rest of the report.

## Future Phases

| Phase | Description |
|-------|-------------|
| **Newsletter email** | Send the digest via email/Resend |
| **RSS/Atom output** | Publish the newsletter as a feed |
| **Auto social posting** | Push social posts to X/Telegram channels |
| **Multi-run trends** | Longitudinal trend analysis across reports |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | ≥ 2.32.0 | HTTP client |
| `python-dotenv` | ≥ 1.0.1 | Load .env configuration |
| `beautifulsoup4` | ≥ 4.12.3 | HTML text extraction |
| `google-genai` | ≥ 1.0.0 | Gemini analysis |
