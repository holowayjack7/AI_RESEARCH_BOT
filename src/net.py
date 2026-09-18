"""Shared HTTP utilities: rate limiting, retries with exponential backoff.

All outbound HTTP in the bot should go through this module so that
rate limits, timeouts, and retry policies are applied consistently.

Retry policy:
- Network-level errors (DNS, timeouts, connection resets): retried
- Retryable HTTP statuses (429/5xx): retried, honoring Retry-After
  (header or, for Telegram, the JSON body's "retry_after" / "parameters")
- Permanent HTTP errors (4xx except 429): raised immediately, no retry
"""

import logging
import threading
import time

import requests

logger = logging.getLogger("ai_research_bot")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Thread-safe limiter enforcing a minimum interval between requests."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        """Block until the next request is allowed."""
        if self.min_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            delay = self._next_time - now
            if delay > 0:
                time.sleep(delay)
            self._next_time = time.monotonic() + self.min_interval


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """Process-wide rate limiter configured from RATE_LIMIT_SECONDS."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            from config import RATE_LIMIT_SECONDS
            _limiter = RateLimiter(RATE_LIMIT_SECONDS)
        return _limiter


def _retry_after_seconds(response: requests.Response) -> float | None:
    """Extract a Retry-After hint from a response, if any.

    Checks the Retry-After header first, then JSON error bodies that
    carry a numeric retry_after (Telegram's 429 responses) or a
    parameters.retry_after object.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass

    try:
        body = response.json()
    except Exception:
        return None

    if isinstance(body, dict):
        for candidate in (
            body.get("retry_after"),
            (body.get("parameters") or {}).get("retry_after")
            if isinstance(body.get("parameters"), dict)
            else None,
        ):
            try:
                return max(1.0, float(candidate))
            except (TypeError, ValueError):
                continue

    return None


def _backoff_from_response(response: requests.Response, attempt: int) -> float:
    """Honor Retry-After if present, else exponential backoff."""
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return retry_after
    return min(2.0 ** attempt, 30.0)


def _network_backoff(attempt: int) -> float:
    """Exponential backoff for network-level errors."""
    return min(2.0 ** attempt, 30.0)


def _request_with_retries(method: str, url: str, max_attempts: int = 3, **kwargs):
    """Run an HTTP request with rate limiting and retries.

    - Retries on network errors and retryable statuses (429/5xx)
    - Raises permanent HTTP errors (4xx except 429) immediately
    - Raises the last error after max_attempts failures
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        get_limiter().wait()

        try:
            response = requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            # Network-level error: retryable
            last_exc = e
            if attempt < max_attempts:
                sleep_for = _network_backoff(attempt)
                logger.warning(
                    f"HTTP {method} {url} network error "
                    f"(attempt {attempt}/{max_attempts}): {e} — "
                    f"retrying in {sleep_for:.0f}s"
                )
                time.sleep(sleep_for)
            continue

        if response.status_code in RETRYABLE_STATUS:
            if attempt < max_attempts:
                sleep_for = _backoff_from_response(response, attempt)
                logger.warning(
                    f"HTTP {method} {url} status {response.status_code} "
                    f"(attempt {attempt}/{max_attempts}) — "
                    f"retrying in {sleep_for:.0f}s"
                )
                time.sleep(sleep_for)
                continue
            last_exc = requests.HTTPError(
                f"{response.status_code} for {url}", response=response
            )
            break

        # Permanent HTTP errors (4xx): raise immediately, no retry
        try:
            response.raise_for_status()
        except requests.HTTPError:
            # No retry policy can fix a permanent error; re-raise as-is
            raise
        return response

    logger.error(f"HTTP {method} {url} failed after {max_attempts} attempts: {last_exc}")
    if last_exc is None:
        # Defensive: the loop can only exhaust via retry paths that set
        # last_exc, but a clear error beats leaking a bare None.
        last_exc = requests.HTTPError(
            f"{method} {url} failed after {max_attempts} attempts"
        )
    raise last_exc


def http_get(url: str, *, params=None, headers=None, timeout: int = 20,
             allow_redirects: bool = True, max_attempts: int = 3) -> requests.Response:
    """Rate-limited GET with retries."""
    return _request_with_retries(
        "GET", url,
        params=params, headers=headers, timeout=timeout,
        allow_redirects=allow_redirects,
        max_attempts=max_attempts,
    )


def http_post(url: str, payload: dict, *, timeout: int = 20,
              max_attempts: int = 3) -> requests.Response:
    """Rate-limited POST with retries (returns the raw Response)."""
    return _request_with_retries(
        "POST", url,
        json=payload, timeout=timeout,
        max_attempts=max_attempts,
    )


def http_post_json(url: str, payload: dict, *, timeout: int = 20,
                   max_attempts: int = 3) -> dict:
    """Rate-limited POST returning parsed JSON, with retries."""
    response = http_post(url, payload, timeout=timeout, max_attempts=max_attempts)
    return response.json()
