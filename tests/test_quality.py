"""Unit tests for the pipeline quality gate (pure logic, no database)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pipeline"))
import check_quality as q  # noqa: E402

NOW = datetime(2026, 11, 4, 12, 1, tzinfo=timezone.utc)
FRESH = NOW - timedelta(minutes=16)


def test_healthy_symbol_passes():
    assert q.evaluate(("BTCUSDT", FRESH, 17279, 17279, FRESH - timedelta(hours=1)), NOW) == []


def test_stale_candles_fail():
    old = NOW - timedelta(hours=3)
    problems = q.evaluate(("BTCUSDT", old, 100, 100, old - timedelta(hours=1)), NOW)
    assert len(problems) == 1 and "min old" in problems[0]


def test_missing_gold_rows_fail():
    problems = q.evaluate(("SOLUSDT", FRESH, 100, 90, FRESH - timedelta(hours=1)), NOW)
    assert any("gold_features has 90" in p for p in problems)


def test_lagging_or_missing_outcomes_fail():
    assert q.evaluate(("BTCUSDT", FRESH, 100, 100, None), NOW)
    assert q.evaluate(("BTCUSDT", FRESH, 100, 100, FRESH - timedelta(hours=5)), NOW)


def test_empty_symbol_fails():
    assert q.evaluate(("ETHUSDT", None, 0, 0, None), NOW) == ["ETHUSDT: no candles in fact_kline"]


def test_message_shows_utc_even_for_local_timestamps():
    berlin = timezone(timedelta(hours=1))
    old_local = (NOW - timedelta(hours=3)).astimezone(berlin)        # driver may return session time zone
    problems = q.evaluate(("BTCUSDT", old_local, 100, 100, old_local - timedelta(hours=1)), NOW)
    assert "09:01 UTC" in problems[0]
