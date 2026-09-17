"""AI Research Bot — Main Entry Point.

Autonomous AI research & news agency agent: collects AI news and
research papers (Tavily web search, arXiv, Hugging Face Daily Papers),
analyzes them with Gemini, publishes Markdown/JSON/newsletter/social
exports, and delivers a digest to Telegram.

Usage:
    python main.py                  # full run (requires API keys)
    python main.py --simulate      # offline end-to-end simulation
    python main.py --check-telegram # verify Telegram delivery setup
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import config
from src.deliver import (
    build_delayed_message,
    get_telegram_me,
    send_telegram,
)
from src.logger import setup_logging
from src.pipeline import run_pipeline, run_simulated_pipeline


STATE_FILE = config.STATE_FILE


def load_state() -> dict:
    """Load persistent state (seen URLs, events, reports, etc.)."""
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)

    if not os.path.exists(STATE_FILE):
        return {
            "seen_urls": [],
            "seen_event_ids": [],
            "events": [],
            "topics": {},
            "reports": [],
            "learning_queue": [],
            "last_run": None,
        }

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "seen_urls": [],
            "seen_event_ids": [],
            "events": [],
            "topics": {},
            "reports": [],
            "learning_queue": [],
            "last_run": None,
        }


def save_state(state: dict):
    """Save persistent state to disk (atomically)."""
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    temp = STATE_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(temp, STATE_FILE)


def record_run(state: dict) -> None:
    """Stamp the execution timestamp and persist state.

    Called on EVERY exit path — delivered, export-only quiet day, or
    crash — so state.json always shows when the bot last executed.
    Best-effort: a write failure must never mask the run's outcome.
    """
    try:
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
    except Exception as e:
        logging.getLogger("ai_research_bot").warning(
            f"Could not record run timestamp: {e}"
        )


def check_telegram() -> int:
    """Verify Telegram credentials by calling getMe and sending a test note.

    Exit codes: 0 = token+chat work, 1 = misconfigured/unreachable.
    Helps debug delivery problems without running the whole pipeline.
    """
    logger = setup_logging(
        level=config.LOG_LEVEL,
        structured=False,
        log_dir=config.DATA_DIR,
    )

    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set "
            "(see .env.example)"
        )
        return 1

    try:
        me = get_telegram_me(config.TELEGRAM_BOT_TOKEN)
        logger.info(
            f"✅ Bot token valid: @{me.get('username', '?')} "
            f"({me.get('first_name', '?')})"
        )
    except Exception as e:
        logger.error(f"❌ Bot token invalid or unreachable: {e}")
        return 1

    try:
        ok = send_telegram(
            "✅ <b>AI Research Bot</b> — test message. "
            "Delivery is configured correctly.",
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID,
        )
        if ok:
            logger.info(
                "✅ Test message delivered to chat — check your Telegram"
            )
            return 0
        logger.error(
            "❌ Could not deliver to the chat. Most common cause: the "
            "bot has never received a message from you. Open your bot "
            "in Telegram, press Start (or send /start), then retry."
        )
        return 1
    except Exception as e:
        logger.error(f"❌ Test send failed: {e}")
        return 1


def _notify_run_failure(error: Exception) -> None:
    """Best-effort Telegram notice when a run fails or crashes.

    The user expects a daily message: if the pipeline dies (e.g. the
    analysis provider is overloaded), send a short delay notice
    instead of silence. Never raises.
    """
    log = logging.getLogger("ai_research_bot")
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return
    try:
        message = build_delayed_message(f"Reason: {error}")
        if send_telegram(
            message, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
        ):
            log.info("Delay notice delivered to Telegram")
    except Exception as notify_error:
        log.warning(f"Could not deliver delay notice: {notify_error}")


def main() -> int:
    """Run the AI Research Bot.

    Returns a process exit code: 0 = delivered (or export-only quiet
    day), 1 = genuine failure. GitHub Actions relies on this to show
    a red run when the report did not go out.
    """
    logger = setup_logging(
        level=config.LOG_LEVEL,
        structured=config.STRUCTURED_LOGS,
        log_dir=config.DATA_DIR,
    )

    logger.info("AI Research Bot starting")

    # Load state
    state = load_state()

    # Telegram delivery needs BOTH a token and a chat id. A partial
    # configuration would silently skip delivery, so fail fast instead.
    telegram_configured = bool(
        config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID
    )
    if bool(config.TELEGRAM_BOT_TOKEN) != bool(config.TELEGRAM_CHAT_ID):
        logger.warning(
            "Telegram config incomplete: "
            f"TELEGRAM_BOT_TOKEN={'set' if config.TELEGRAM_BOT_TOKEN else 'MISSING'}, "
            f"TELEGRAM_CHAT_ID={'set' if config.TELEGRAM_CHAT_ID else 'MISSING'} — "
            "delivery will be SKIPPED. Set both in .env."
        )

    try:
        config.validate_config(require_telegram=telegram_configured)
        logger.info("Configuration loaded")
    except RuntimeError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    if telegram_configured:
        # Diagnose delivery problems (bad token, bot never started by
        # the user) BEFORE spending the whole pipeline run on them.
        try:
            me = get_telegram_me(config.TELEGRAM_BOT_TOKEN)
            logger.info(
                f"Telegram bot verified: @{me.get('username', '?')}"
            )
        except Exception as e:
            logger.warning(
                f"Telegram bot check failed: {e} — delivery may fail; "
                "continuing (report files are still exported)"
            )

    logger.info("Pipeline initialized")
    logger.info("Research pipeline started")

    # Run the full pipeline
    try:
        success = run_pipeline(
            tavily_key=config.TAVILY_API_KEY,
            gemini_key=config.GEMINI_API_KEY,
            telegram_token=config.TELEGRAM_BOT_TOKEN if telegram_configured else "",
            telegram_chat_id=config.TELEGRAM_CHAT_ID if telegram_configured else "",
            state=state,
            gemini_model=config.GEMINI_MODEL,
        )

        # Record the run even when nothing was delivered
        # (no candidates / no important events), same as the monolith.
        record_run(state)

        if success:
            logger.info("Research pipeline finished successfully")
        elif telegram_configured:
            # With Telegram configured, False means the report did not
            # go out (publish/send/heartbeat failure): surface it as a
            # failed run, never a silent green check in Actions.
            logger.error("Research pipeline finished WITHOUT delivery — failing run")
            _notify_run_failure(
                RuntimeError("delivery did not complete (see run logs)")
            )
            return 1
        else:
            # Export-only run without Telegram: nothing to deliver is normal
            logger.info("Research pipeline finished (no delivery; export-only run)")
            return 0

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Even a crashed run must update the execution timestamp so
        # state.json always reflects the latest execution attempt.
        record_run(state)
        _notify_run_failure(e)
        return 1

    logger.info("AI Research Bot finished")
    return 0


def run_simulation() -> None:
    """Offline end-to-end run: fixture data, real pipeline code."""
    logger = setup_logging(
        level=config.LOG_LEVEL,
        structured=config.STRUCTURED_LOGS,
        log_dir=config.DATA_DIR,
    )

    logger.info("AI Research Bot — SIMULATION MODE")

    state = load_state()

    try:
        success = run_simulated_pipeline(state)
        record_run(state)
        logger.info(
            "Simulation finished successfully"
            if success
            else "Simulation finished (nothing new to publish)"
        )
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Research Bot")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Offline end-to-end simulation (no network, no API keys)",
    )
    parser.add_argument(
        "--check-telegram",
        action="store_true",
        help="Verify Telegram bot token + chat and send a test message",
    )
    args = parser.parse_args()

    try:
        if args.check_telegram:
            sys.exit(check_telegram())
        elif args.simulate:
            run_simulation()
        else:
            sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise
