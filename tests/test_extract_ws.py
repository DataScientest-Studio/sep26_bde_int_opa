"""Unit tests for extract_ws.py (no internet and no database needed)."""
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

# Make src/data importable, like tests/test_transform.py does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

import extract_ws  # noqa: E402

AGREED_FIELDS = {"symbol", "interval", "source", "ingested_at", "payload"}  # see README


def message(closed):
    """A combined-stream kline message exactly as Binance sends it."""
    return {"stream": "btcusdt@kline_1m", "data": {
        "e": "kline", "E": 1760000460005, "s": "BTCUSDT",
        "k": {"t": 1760000400000, "T": 1760000459999, "s": "BTCUSDT", "i": "1m", "f": 100, "L": 200,
              "o": "121300.00", "c": "121320.40", "h": "121335.00", "l": "121295.10", "v": "6.512",
              "n": 1023, "x": closed, "q": "790010.33", "V": "3.1", "Q": "376100.20", "B": "0"}}}


class FakeCollection:
    """Stands in for MongoDB: just remembers what was inserted."""

    def __init__(self):
        self.saved = []

    def insert_one(self, doc):
        self.saved.append(doc)


def test_document_follows_the_agreed_format():
    doc = extract_ws.make_document(message(closed=True))
    assert set(doc) == AGREED_FIELDS
    assert (doc["symbol"], doc["interval"], doc["source"]) == ("BTCUSDT", "1m", "websocket")
    assert doc["payload"] == message(closed=True)     # the whole message, unchanged


def test_stream_url_uses_lower_case_names():
    expected = "/".join(f"{s.lower()}@kline_{extract_ws.INTERVAL}" for s in extract_ws.SYMBOLS)
    assert extract_ws.stream_url().endswith(f"/stream?streams={expected}")
    assert not any(symbol in extract_ws.stream_url() for symbol in extract_ws.SYMBOLS)  # no upper-case names


def test_only_finished_candles_are_saved():
    collection = FakeCollection()
    assert extract_ws.save_if_finished(collection, json.dumps(message(closed=False))) is False
    assert extract_ws.save_if_finished(collection, json.dumps(message(closed=True))) is True
    assert len(collection.saved) == 1


def test_transform_can_read_the_documents():
    transform = pytest.importorskip("transform")      # the contract test runs once transform.py exists
    if not hasattr(transform, "parse_document"):
        pytest.skip("transform.py is not implemented yet")
    row = transform.parse_document(extract_ws.make_document(message(closed=True)))
    assert row["open_time"] == datetime(2025, 10, 9, 9, 0, tzinfo=timezone.utc)
    assert row["close"] == Decimal("121320.40")
