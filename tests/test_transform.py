"""Unit tests for the parsing logic in transform.py (no database needed)."""
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

# Make src/data importable and stub the DB drivers, so the tests run without them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
for mod in ("psycopg2", "psycopg2.extras", "pymongo", "dotenv"):
    sys.modules.setdefault(mod, MagicMock())

import transform  # noqa: E402

REST_DOC = {
    "symbol": "BTCUSDT", "interval": "15m", "source": "rest",
    "payload": [1760000400000, "121300.00", "121450.10", "121250.00", "121400.50", "85.123",
                1760001299999, "10331234.56", 15234, "40.1", "4867000.12", "0"],
}


def ws_doc(closed):
    return {
        "symbol": "BTCUSDT", "interval": "1m", "source": "websocket",
        "payload": {"stream": "btcusdt@kline_1m", "data": {"k": {
            "t": 1760000400000, "T": 1760000459999, "o": "1", "h": "2", "l": "0.5", "c": "1.5",
            "v": "10", "q": "15", "n": 7, "x": closed}}},
    }


def test_rest_parsing():
    row = transform.parse_document(REST_DOC)
    assert row["open_time"] == datetime(2025, 10, 9, 9, 0, tzinfo=timezone.utc)
    assert row["close"] == Decimal("121400.50")
    assert row["trades"] == 15234
    assert row["symbol"] == "BTCUSDT"


def test_ws_closed_candle_is_kept():
    row = transform.parse_document(ws_doc(closed=True))
    assert row["high"] == Decimal("2")
    assert row["trades"] == 7


def test_ws_open_candle_is_skipped():
    assert transform.parse_document(ws_doc(closed=False)) is None


def test_ws_kline_found_in_any_shape():
    k = {"t": 1}
    assert transform.find_ws_kline({"data": {"k": k}}) == k
    assert transform.find_ws_kline({"k": k}) == k
    assert transform.find_ws_kline(k) == k


def test_rest_candle_not_finished_yet_is_skipped():
    before_close = datetime(2025, 10, 9, 9, 5, tzinfo=timezone.utc)
    assert transform.parse_document(REST_DOC, now=before_close) is None
