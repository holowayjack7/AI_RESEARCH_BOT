"""Logging configuration for the AI Research Bot.

Console logging (human-readable) plus an optional structured JSONL
run log under DATA_DIR for machine-readable history.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonlHandler(logging.Handler):
    """Append each record as one JSON line to a structured log file."""

    def __init__(self, path: str, level: int = logging.INFO):
        super().__init__(level)
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["error"] = self.format_exception(record.exc_info)

            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)

    @staticmethod
    def format_exception(exc_info) -> str:
        import traceback
        return "".join(traceback.format_exception(*exc_info)).strip()


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_dir: str = "data",
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        structured: Also write JSONL events to <log_dir>/logs/bot.jsonl.
        log_dir: Base directory for structured logs.

    Returns:
        Configured logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("ai_research_bot")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if structured:
        jsonl_path = os.path.join(log_dir, "logs", "bot.jsonl")
        try:
            logger.addHandler(JsonlHandler(jsonl_path, numeric_level))
        except Exception:
            # Structured logging must never break the bot
            logger.warning("Structured logging unavailable — continuing console-only")

    return logger
