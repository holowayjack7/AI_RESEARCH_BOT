"""Configuration module for AI Research Bot.

Loads environment variables from .env file using python-dotenv.
Only GEMINI_API_KEY is strictly required: Tavily adds web search,
arXiv/Hugging Face are keyless, and Telegram credentials enable
chat delivery.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with an optional default.

    Empty or whitespace-only values are treated as unset. This matters
    in GitHub Actions: a missing secret (e.g. GEMINI_MODEL) is injected
    as an empty string, which would otherwise override the default and
    break the pipeline.
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_env_bool(key: str, default: bool) -> bool:
    """Get a boolean environment variable."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_env_int(key: str, default: int) -> int:
    """Get an integer environment variable (falls back on bad values)."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# === API Keys ===
TAVILY_API_KEY = get_env("TAVILY_API_KEY")
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")

# === Model Config ===
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-3.6-flash")
# Comma-separated fallback models tried in order when the primary
# model keeps failing (e.g. 503 "high demand" capacity spikes)
GEMINI_MODEL_FALLBACKS = [
    m.strip() for m in get_env(
        "GEMINI_MODEL_FALLBACKS", "gemini-2.5-flash"
    ).split(",") if m.strip()
]

# === Feature flags: sources ===
ENABLE_ARXIV = get_env_bool("ENABLE_ARXIV", True)
ENABLE_HF_PAPERS = get_env_bool("ENABLE_HF_PAPERS", True)

# === Sources: papers ===
ARXIV_MAX_RESULTS = get_env_int("ARXIV_MAX_RESULTS", 15)
HF_MAX_RESULTS = get_env_int("HF_MAX_RESULTS", 15)
PAPER_MAX_AGE_DAYS = get_env_int("PAPER_MAX_AGE_DAYS", 3)

# === Search Config ===
MAX_RESULTS_PER_QUERY = 8
MAX_CANDIDATES_FOR_RESEARCH = 24
MAX_ARTICLE_CHARS = 7000

# === Gemini analysis budget ===
# Per-candidate content cap sent to Gemini (the fetcher keeps more for
# context, but the prompt must fit the free-tier per-minute quota)
GEMINI_CONTENT_CHARS = get_env_int("GEMINI_CONTENT_CHARS", 3000)
# Total character budget for source content in the prompt
# (~4 chars/token -> ~50K tokens, far under the 250K-token free-tier
# per-minute input quota for gemini-3.6-flash)
GEMINI_PROMPT_CHAR_BUDGET = get_env_int("GEMINI_PROMPT_CHAR_BUDGET", 200000)
# Analysis attempts (retries honor Gemini's "retry in Xs" hints)
GEMINI_MAX_ATTEMPTS = get_env_int("GEMINI_MAX_ATTEMPTS", 4)

# === Quality Thresholds ===
MIN_SOURCE_QUALITY = get_env_int("MIN_SOURCE_QUALITY", 7)
MIN_IMPORTANCE = get_env_int("MIN_IMPORTANCE", 6)
MIN_RELEVANCE = get_env_int("MIN_RELEVANCE", 6)
MIN_ACTIONABILITY = get_env_int("MIN_ACTIONABILITY", 5)
MIN_CONFIDENCE = get_env_int("MIN_CONFIDENCE", 65)

# === Telegram Config ===
TELEGRAM_CHUNK_SIZE = 3800

# === HTTP / Rate Limit Config ===
REQUEST_TIMEOUT = 20
try:
    RATE_LIMIT_SECONDS = float(get_env("RATE_LIMIT_SECONDS", "1.0"))
except ValueError:
    RATE_LIMIT_SECONDS = 1.0
if RATE_LIMIT_SECONDS < 0:
    RATE_LIMIT_SECONDS = 0.0

# === Logging / Publishing ===
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
STRUCTURED_LOGS = get_env_bool("STRUCTURED_LOGS", True)
DATA_DIR = get_env("DATA_DIR", "data")
REPORTS_SUBDIR = get_env("REPORTS_SUBDIR", "reports")


def validate_config(require_telegram: bool = True) -> None:
    """Check that required environment variables are set.

    Only GEMINI_API_KEY is strictly required (analysis always runs).
    Tavily is optional — without it the bot still collects from arXiv
    and Hugging Face. Telegram credentials are only required when
    actually delivering to chat.
    """
    missing = []

    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }

    if require_telegram:
        required["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN
        required["TELEGRAM_CHAT_ID"] = TELEGRAM_CHAT_ID

    for name, value in required.items():
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )
