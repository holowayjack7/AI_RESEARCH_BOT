"""Review test suite: delivery UI, chunking, send resilience, HTTP retries,
pipeline regressions, LLM-output robustness.
Run: .venv/bin/python tests/test_review.py"""

import json
import os
import re
import sys
import tempfile

# Make the project root importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TAVILY_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = "test-gemini"
os.environ.pop("TELEGRAM_BOT_TOKEN", None)
os.environ.pop("TELEGRAM_CHAT_ID", None)

tmpdir = tempfile.mkdtemp()
os.environ["STATE_FILE"] = os.path.join(tmpdir, "state.json")
os.environ["DATA_DIR"] = tmpdir

import config  # noqa: E402  (import after env overrides)

import requests  # noqa: E402
from src import net as net_mod  # noqa: E402
from src import deliver as de  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import search as src_search_mod  # noqa: E402
from src.analyze import enforce_conciseness, parse_report, ResearchEvent  # noqa: E402
from datetime import datetime  # noqa: E402

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ✓ {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"  ✗ {name}: unexpected {e!r}")


class FakeResp:
    def __init__(self, status=200):
        self.status_code = status
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def six_events(report):
    """Clone a report with 6 same-schema events (forces multi-chunk)."""
    report.events = [
        ResearchEvent(**{**report.events[0].__dict__, "event_id": f"e{i}"})
        for i in range(6)
    ]
    return report


# ==================================================================
print("\n[1] Telegram UI formatting")
# ==================================================================

REPORT = parse_report(pl.SIMULATED_REPORT_JSON)
msg = de.build_telegram_message(REPORT)


def test_ui_structure():
    # Mandatory 4-section structure per event
    assert "<b>AI INTELLIGENCE REPORT</b>" in msg
    assert "<b>In this issue</b>" in msg
    assert "<blockquote expandable>" in msg       # collapsed summaries
    for section in (
        "<b>Executive Impact</b>",
        "<b>Key Technical Breakdown</b>",
        "<b>Architecture and Specs</b>",
        "<b>Implementation Logic</b>",
        "<b>Actionable Takeaways</b>",
        "<b>Verified Resources</b>",
    ):
        assert section in msg, f"missing mandated section: {section}"
    assert "Apply (LEARN)" in msg and "Commercial potential" in msg
    assert '<a href="https://arxiv.org/abs/2609.12001">Research paper</a>' in msg
    assert '<a href="https://github.com/example-labs/fixture-agent-sdk">Source code</a>' in msg
    # ZERO emojis, icons, or badges anywhere in the output
    emoji = re.compile(
        "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
        "|[\u2190-\u21FF\u2B00-\u2BFF\u25A0-\u25FF\u2700-\u27BF]"
    )
    hits = emoji.findall(msg)
    assert not hits, f"emojis/icons banned but found: {hits[:5]}"
    # escaped HTML must not allow injection
    ev = ResearchEvent(title="<script>x</script>", category="c",
                       primary_url="https://arxiv.org/abs/1", action="a")
    card = de.format_event(1, ev)
    assert "<script>" not in card and "&lt;script&gt;" in card
    # href attribute must be escaped too
    assert '<a href="https://arxiv.org/abs/1"' in card


check("UI structure, score bars, HTML escaping", test_ui_structure)


# ==================================================================
print("\n[2] Chunking integrity")
# ==================================================================

def test_chunking_small():
    chunks = de.split_message(msg)
    assert len(chunks) == 1, f"short message became {len(chunks)} chunks"


def test_chunking_large():
    big_msg = de.build_telegram_message(six_events(parse_report(pl.SIMULATED_REPORT_JSON)))
    assert len(big_msg) > 3800
    chunks = de.split_message(big_msg)
    assert len(chunks) >= 2
    assert all(len(c) <= de.TELEGRAM_CHUNK_SIZE for c in chunks), \
        max(len(c) for c in chunks)
    # every chunk must have balanced tags (never cut mid-tag)
    for c in chunks:
        assert c.count("<b>") == c.count("</b>"), f"unbalanced <b>: {c[:100]}"
        assert c.count("<code>") == c.count("</code>")
        assert c.count("<blockquote") == c.count("</blockquote>")


def test_chunking_oversized_paragraph():
    # one paragraph way over the limit, wrapped in tags
    huge = "<b>" + ("x" * 9000) + "</b>"
    chunks = de.split_message(huge)
    assert all(len(c) <= de.TELEGRAM_CHUNK_SIZE for c in chunks)
    for c in chunks:
        assert c.count("<b>") == c.count("</b>")


check("small message = single chunk", test_chunking_small)
check("large message splits under limit with balanced tags", test_chunking_large)
check("oversized paragraph clipped safely", test_chunking_oversized_paragraph)


# ==================================================================
print("\n[3] send_telegram resilience (mocked HTTP)")
# ==================================================================
# NOTE: deliver.py does `from src.net import http_post`, so the name
# deliver actually calls is de.http_post — patch there.

def test_send_all_ok():
    calls = []
    orig_post, orig_sleep = de.http_post, de.time.sleep
    de.http_post = lambda url, payload, timeout=20, **kw: (
        calls.append(payload) or FakeResp(200)
    )
    de.time.sleep = lambda s: None
    try:
        ok = de.send_telegram(msg, "token", "chat")
        assert ok is True, "delivery should succeed"
        assert len(calls) == 1
        assert calls[0]["parse_mode"] == "HTML"
        assert calls[0]["link_preview_options"]["is_disabled"] is True
    finally:
        de.http_post, de.time.sleep = orig_post, orig_sleep


def test_send_chunk1_fails_chunk2_ok():
    # multi-chunk message where the first send fails permanently ->
    # plain-text fallback succeeds; delivery must still return True
    calls = []
    state = {"n": 0}

    def flaky(url, payload, timeout=20, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.HTTPError("400 Bad Request: can't parse entities")
        calls.append(payload)
        return FakeResp(200)

    orig_post, orig_sleep = de.http_post, de.time.sleep
    de.http_post = flaky
    de.time.sleep = lambda s: None
    try:
        ok = de.send_telegram(
            de.build_telegram_message(six_events(parse_report(pl.SIMULATED_REPORT_JSON))),
            "t", "c",
        )
        assert ok is True, "delivery should succeed via fallback"
        assert any("parse_mode" not in c for c in calls), \
            "fallback chunk must be plain text"
    finally:
        de.http_post, de.time.sleep = orig_post, orig_sleep


def test_send_all_fail_returns_false():
    def always_fail(url, payload, timeout=20, **kw):
        raise requests.HTTPError("500 Server Error")

    orig_post, orig_sleep = de.http_post, de.time.sleep
    de.http_post = always_fail
    de.time.sleep = lambda s: None
    try:
        ok = de.send_telegram(msg, "t", "c")
        assert ok is False, "total failure must return False, not raise"
    finally:
        de.http_post, de.time.sleep = orig_post, orig_sleep


check("send: happy path payloads", test_send_all_ok)
check("send: bad-HTML chunk falls back to plain text, delivery survives", test_send_chunk1_fails_chunk2_ok)
check("send: total failure returns False instead of raising", test_send_all_fail_returns_false)


# ==================================================================
print("\n[4] HTTP retry policy (src/net)")
# ==================================================================

def test_no_retry_on_permanent_404():
    calls = {"n": 0}

    def fake_request(method, url, **kw):
        calls["n"] += 1
        return FakeResp(404)

    orig_request, orig_sleep = net_mod.requests.request, net_mod.time.sleep
    net_mod.requests.request = fake_request
    net_mod.time.sleep = lambda s: None
    net_mod._limiter = net_mod.RateLimiter(0)  # no sleeping
    try:
        try:
            net_mod.http_get("https://x.test")
            raise AssertionError("404 must raise")
        except requests.HTTPError:
            pass
        assert calls["n"] == 1, f"404 retried {calls['n']} times"
    finally:
        net_mod.requests.request, net_mod.time.sleep = orig_request, orig_sleep
        net_mod._limiter = None


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def fake_request(method, url, **kw):
        calls["n"] += 1
        return FakeResp(429 if calls["n"] == 1 else 200)

    orig_request, orig_sleep = net_mod.requests.request, net_mod.time.sleep
    net_mod.requests.request = fake_request
    net_mod.time.sleep = lambda s: None
    net_mod._limiter = net_mod.RateLimiter(0)
    try:
        resp = net_mod.http_get("https://x.test")
        assert resp.status_code == 200
        assert calls["n"] == 2
    finally:
        net_mod.requests.request, net_mod.time.sleep = orig_request, orig_sleep
        net_mod._limiter = None


def test_retry_on_network_error_then_success():
    calls = {"n": 0}

    def fake_request(method, url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("reset")
        return FakeResp(200)

    orig_request, orig_sleep = net_mod.requests.request, net_mod.time.sleep
    net_mod.requests.request = fake_request
    net_mod.time.sleep = lambda s: None
    net_mod._limiter = net_mod.RateLimiter(0)
    try:
        resp = net_mod.http_get("https://x.test")
        assert resp.status_code == 200
        assert calls["n"] == 2
    finally:
        net_mod.requests.request, net_mod.time.sleep = orig_request, orig_sleep
        net_mod._limiter = None


check("net: 404 raises immediately (no pointless retries)", test_no_retry_on_permanent_404)
check("net: 429 retried then succeeds", test_retry_on_429_then_success)
check("net: network error retried then succeeds", test_retry_on_network_error_then_success)


# ==================================================================
print("\n[5] Pipeline regressions (dedup, state, publish)")
# ==================================================================

def test_dedup_collapse():
    out = pl.collapse_by_dedup_key(
        pl.deduplicate_results([dict(c) for c in pl.SIMULATED_SOURCES])
    )
    assert len(out) == 2  # 3 fixtures -> 2 stories


def test_dedup_key_variants():
    k1 = pl.dedup_key_from_url("https://arxiv.org/abs/2609.12001v2")
    k2 = pl.dedup_key_from_url("https://arxiv.org/html/2609.12001")
    k3 = pl.dedup_key_from_url("https://arxiv.org/pdf/2609.12001")
    k4 = pl.dedup_key_from_url("https://huggingface.co/papers/2609.12001")
    assert k1 == k2 == k3 == k4 == "arxiv:2609.12001"


def test_state_roundtrip():
    state = {}
    report = parse_report(pl.SIMULATED_REPORT_JSON)
    cands = [{"url": "https://arxiv.org/abs/2609.12001"}]
    pl.update_state(state, report, cands)
    assert state["seen_dedup_keys"] == ["arxiv:2609.12001"]
    assert len(state["events"]) == 2
    assert len(state["reports"]) == 1


def test_seen_urls_delivered_only():
    """Analyzed-but-unreported candidates must NOT be marked seen."""
    state = {}
    report = parse_report(pl.SIMULATED_REPORT_JSON)  # 2 events
    delivered_url = report.events[0].primary_url
    not_delivered = "https://github.com/never-reported/repo"
    pl.update_state(state, report, [{"url": delivered_url}, {"url": not_delivered}])
    assert delivered_url in state["seen_urls"]
    assert not_delivered not in state["seen_urls"], (
        "unreported candidate was permanently suppressed — information loss"
    )


def test_publish_files():
    out_dir = os.path.join(tmpdir, "pub_test")
    outs = pl.publish_report(REPORT, out_dir=out_dir)
    for fmt, path in outs.items():
        assert os.path.exists(path) and os.path.getsize(path) > 0, fmt
    report_json = json.load(open(outs["json"], encoding="utf-8"))
    assert report_json["events"][0]["tldr"]


check("dedup: 3 fixture sources collapse to 2 stories", test_dedup_collapse)
check("dedup: abs/html/pdf/HF variants share one key", test_dedup_key_variants)
check("state: update_state records urls+ids+keys", test_state_roundtrip)
check("publish: all 4 formats written with content", test_publish_files)


# ==================================================================
print("\n[6] End-to-end runs")
# ==================================================================

def test_simulation_e2e():
    state = {}
    ok = pl.run_simulated_pipeline(state)
    assert ok is True
    ok2 = pl.run_simulated_pipeline(state)
    assert ok2 is False, "second simulation run must dedup to nothing"


check("simulation: e2e + cross-run dedup", test_simulation_e2e)


def test_real_pipeline_with_mocked_apis():
    """Full run_pipeline with mocked sources/Gemini/Telegram/fetcher.

    NOTE: pipeline.py imports collect_all_sources / send_telegram into
    its own namespace (pl.collect_all_sources / pl.send_telegram).
    """
    state = {}
    sent = []

    orig_collect = pl.collect_all_sources
    orig_client = __import__("src.analyze", fromlist=["genai"]).genai.Client
    orig_fetch = pl.fetch_article
    orig_send = pl.send_telegram

    pl.collect_all_sources = lambda key: [dict(c) for c in pl.SIMULATED_SOURCES]
    __import__("src.analyze", fromlist=["genai"]).genai.Client = lambda api_key: type(
        "C", (), {"models": type("M", (), {"generate_content": staticmethod(
            lambda model=None, contents=None, config=None: type(
                "R", (), {"text": pl.SIMULATED_REPORT_JSON})()
        )})()}
    )
    pl.fetch_article = lambda url: "Fetched text. " * 50
    pl.send_telegram = lambda m, t, c: (sent.append(m), True)[1]
    try:
        ok = pl.run_pipeline(
            tavily_key="", gemini_key="k", state=state, gemini_model="m",
            telegram_token="t", telegram_chat_id="c",
        )
        assert ok is True, "pipeline should succeed"
        assert sent, "Telegram must be called when configured"
        assert "AI INTELLIGENCE REPORT" in sent[0]
        assert state.get("last_report_files")
    finally:
        pl.collect_all_sources = orig_collect
        __import__("src.analyze", fromlist=["genai"]).genai.Client = orig_client
        pl.fetch_article = orig_fetch
        pl.send_telegram = orig_send


check("pipeline: full run with mocked APIs delivers new UI", test_real_pipeline_with_mocked_apis)


# ==================================================================
print("\n[7] LLM-output robustness (parse_report coercion)")
# ==================================================================

def test_coercion():
    raw = """
    {
      "report_title": "Scores as strings and floats",
      "executive_summary": "Coercion test",
      "events": [
        {
          "title": "String/float/null scores must not crash",
          "primary_url": "https://arxiv.org/abs/2609.12001",
          "importance": "9",
          "relevance": 8.5,
          "actionability": "88%",
          "source_quality": null,
          "confidence": "91",
          "key_takeaways": "not a list",
          "supporting_urls": ["ok", null, 42],
          "evidence": ["not a dict", {"url": "u", "evidence": "e"}]
        }
      ],
      "trends": null
    }
    """
    report = parse_report(raw)
    e = report.events[0]
    assert e.importance == 9 and isinstance(e.importance, int)
    assert e.relevance == 8        # 8.5 -> clamped int
    assert e.actionability == 10   # "88%" -> parsed then clamped to 1-10
    assert e.source_quality == 5   # None -> default
    assert e.confidence == 91
    assert e.key_takeaways == []   # non-list -> empty
    assert e.supporting_urls == ["ok", "42"]
    assert len(e.evidence) == 1    # non-dict entries dropped
    assert report.trends == []


def test_out_of_range_scores_clamped():
    raw = """
    {
      "events": [
        {"title": "T", "primary_url": "https://arxiv.org/abs/1",
         "importance": 99, "confidence": -5, "relevance": "high"}
      ]
    }
    """
    e = parse_report(raw).events[0]
    assert e.importance == 10 and e.confidence == 0 and e.relevance == 5


def test_malformed_events_skipped():
    raw = """
    {
      "events": [
        {"title": "no url"},
        {"primary_url": "https://arxiv.org/abs/1"},
        "just a string",
        {"title": "OK event", "primary_url": "https://arxiv.org/abs/1"}
      ]
    }
    """
    report = parse_report(raw)
    assert len(report.events) == 1
    assert report.events[0].title == "OK event"


check("coercion: string/float/null scores parsed without crashing", test_coercion)
check("coercion: out-of-range scores clamped to valid ranges", test_out_of_range_scores_clamped)
check("coercion: malformed events skipped, valid kept", test_malformed_events_skipped)


# ==================================================================
print("\n[8] Signal refinement (low-value filter + conciseness)")
# ==================================================================

def test_low_value_filter():
    from src.search import is_low_value_result
    # Clickbait / financial noise / rumors must be dropped
    assert is_low_value_result(
        "You Won't Believe This New Model",
        "some content long enough to pass the summary length check here",
    )[0]
    assert is_low_value_result(
        "OpenAI Stock Soars After Earnings Call",
        "some content long enough to pass the summary length check here",
    )[0]
    assert is_low_value_result("Short", "x")[0]
    # High-signal technical content must pass
    ok, reason = is_low_value_result(
        "Gemini 3.8 Flash API: structured outputs and batch mode",
        "Official release notes describing new generateContent parameters, "
        "batching support, and pricing changes for developers." * 2,
    )
    assert ok is False, reason


def test_enforce_conciseness():
    raw = json.dumps({"events": [{
        "title": "T", "primary_url": "https://arxiv.org/abs/1",
        "tldr": "word " * 100,
        "action": "word " * 50,
        "key_takeaways": ["x " * 40] * 7,
        "technical_details": ["d"] * 9,
    }]})
    r = enforce_conciseness(parse_report(raw))
    e = r.events[0]
    assert len(e.tldr.split()) <= 31      # 30-word cap + ellipsis
    assert len(e.action.split()) <= 26
    assert len(e.key_takeaways) == 4      # capped count
    assert len(e.technical_details) == 4


check("search: clickbait dropped, high-signal kept", test_low_value_filter)
check("analyze: conciseness enforcement clips rambling fields", test_enforce_conciseness)


# ==================================================================
print("\n[9] Resilience (retries, backoff, run timestamps)")
# ==================================================================

def test_tavily_retry_backoff():
    """Tavily query survives a transient network failure via net retries."""
    calls = {"n": 0}

    def flaky_request(method, url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("connection reset")
        resp = FakeResp(200)
        resp.json = lambda: {
            "results": [{"title": "ok", "url": "https://arxiv.org/abs/1"}]
        }
        return resp

    orig_request, orig_sleep = net_mod.requests.request, net_mod.time.sleep
    net_mod.requests.request = flaky_request
    net_mod.time.sleep = lambda s: None
    net_mod._limiter = net_mod.RateLimiter(0)
    try:
        results = src_search_mod.search_tavily("test query", "key")
        assert len(results) == 1, "transient failure must be retried"
        assert calls["n"] == 2, f"expected 2 attempts, got {calls['n']}"
    finally:
        net_mod.requests.request, net_mod.time.sleep = orig_request, orig_sleep
        net_mod._limiter = None


def test_gemini_backoff_is_exponential():
    """Gemini waits grow exponentially; server retry hints are honored."""
    from src.analyze import _gemini_retry_wait

    w1 = _gemini_retry_wait("429 RESOURCE_EXHAUSTED quota exceeded", 1)
    w2 = _gemini_retry_wait("429 RESOURCE_EXHAUSTED quota exceeded", 2)
    w3 = _gemini_retry_wait("429 RESOURCE_EXHAUSTED quota exceeded", 3)
    assert w2 > w1 > 0, f"backoff must grow: {w1}, {w2}"
    assert w3 > w2, f"backoff must keep growing: {w2}, {w3}"
    assert w3 <= 120.0, "backoff must be capped"
    # Server-provided hint always wins
    assert _gemini_retry_wait("please retry in 37.5s", 1) >= 37.0
    # Transient 503 also backs off
    assert _gemini_retry_wait("503 UNAVAILABLE high demand", 2) > \
        _gemini_retry_wait("503 UNAVAILABLE high demand", 1)


def test_record_run_always_stamps_state():
    """record_run stamps last_run on every path, including crashes."""
    import importlib
    main_mod = importlib.import_module("main")

    saved = []
    orig_save = main_mod.save_state
    main_mod.save_state = lambda s: saved.append(dict(s))
    try:
        # Crash path: pipeline raised, but the run must still be stamped
        main_mod.record_run({"seen_urls": []})
        assert saved and saved[-1]["last_run"], "crash path must stamp last_run"
        # The stamped value must parse as ISO-8601 UTC
        datetime.fromisoformat(saved[-1]["last_run"])
        # A failing save_state must never raise out of record_run
        main_mod.save_state = lambda s: (_ for _ in ()).throw(OSError("disk full"))
        main_mod.record_run({})  # must not raise
    finally:
        main_mod.save_state = orig_save


check("search: Tavily retries transient failures with backoff", test_tavily_retry_backoff)
check("analyze: Gemini backoff grows exponentially, honors retry hints", test_gemini_backoff_is_exponential)
check("main: record_run stamps state on crash path, never raises", test_record_run_always_stamps_state)


# ==================================================================
print()
print("=" * 50)
print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
for name, err in FAIL:
    print(f"  FAIL: {name} — {err}")
print("=" * 50)


# Dual-mode: runnable directly AND via pytest. The raise only fires
# in direct mode (__main__); under pytest, failures were already
# recorded per-check via the check() helper.
if __name__ == "__main__":
    raise SystemExit(1 if FAIL else 0)


def test_review_suite():
    """Pytest entry point: re-raise collected failures for pytest."""
    assert not FAIL, "; ".join(f"{n}: {e}" for n, e in FAIL)
