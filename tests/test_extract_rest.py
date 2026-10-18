"""Unit tests for extract_rest.py (no internet and no database needed)."""
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

# Make src/data importable, like tests/test_transform.py does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

import extract_rest  # noqa: E402

AGREED_FIELDS = {"symbol", "interval", "source", "ingested_at", "payload"}  # see README

# Two 15-minute klines exactly as /api/v3/klines returns them (12 values each)
KLINES = [
    [1760000400000, "121300.00", "121450.10", "121250.00", "121400.50", "85.123",
     1760001299999, "10331234.56", 15234, "40.1", "4867000.12", "0"],
    [1760001300000, "121400.50", "121500.00", "121350.20", "121480.00", "70.456",
     1760002199999, "8557000.10", 12001, "35.2", "4275000.55", "0"],
]


def test_end_time_stops_before_the_running_candle():
    step = extract_rest.INTERVAL_MS["15m"]
    candle_open = 1760000400000                      # a candle that opened at a 15-minute boundary
    assert extract_rest.last_closed_end(candle_open + 5 * 60_000, step) == candle_open - 1
    assert extract_rest.last_closed_end(candle_open, step) == candle_open - 1


def test_one_document_per_kline_in_the_agreed_format():
    docs = extract_rest.make_documents("BTCUSDT", "15m", KLINES)
    assert len(docs) == len(KLINES)
    for doc, kline in zip(docs, KLINES):
        assert set(doc) == AGREED_FIELDS
        assert (doc["symbol"], doc["interval"], doc["source"]) == ("BTCUSDT", "15m", "rest")
        assert doc["payload"] == kline                # stored exactly as Binance sent it


def test_transform_can_read_the_documents():
    transform = pytest.importorskip("transform")      # the contract test runs once transform.py exists
    if not hasattr(transform, "parse_document"):
        pytest.skip("transform.py is not implemented yet")
    doc = extract_rest.make_documents("BTCUSDT", "15m", KLINES[:1])[0]
    row = transform.parse_document(doc, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert row["open_time"] == datetime(2025, 10, 9, 9, 0, tzinfo=timezone.utc)
    assert row["close"] == Decimal("121400.50")
