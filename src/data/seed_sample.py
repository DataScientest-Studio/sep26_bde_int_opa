"""Inserts a few sample bronze documents into Mongo so transform.py can be tested
before the real extractors are ready. Safe to delete later.

Run from the repo root:
    python src/data/seed_sample.py
"""
from datetime import datetime, timezone

from db import get_mongo_collection

NOW = datetime.now(timezone.utc)

SAMPLE_DOCS = [
    # REST: kline arrays exactly as /api/v3/klines returns them (15m candles)
    {
        "symbol": "BTCUSDT", "interval": "15m", "source": "rest", "ingested_at": NOW,
        "payload": [1760000400000, "121300.00", "121450.10", "121250.00", "121400.50", "85.123",
                    1760001299999, "10331234.56", 15234, "40.1", "4867000.12", "0"],
    },
    {
        "symbol": "BTCUSDT", "interval": "15m", "source": "rest", "ingested_at": NOW,
        "payload": [1760001300000, "121400.50", "121500.00", "121350.20", "121480.00", "70.456",
                    1760002199999, "8557000.10", 12001, "35.2", "4275000.55", "0"],
    },
    {
        "symbol": "SOLUSDT", "interval": "15m", "source": "rest", "ingested_at": NOW,
        "payload": [1760000400000, "222.10", "222.95", "221.80", "222.60", "15234.5",
                    1760001299999, "3388000.25", 8450, "7600.1", "1690000.40", "0"],
    },
    # WebSocket: combined-stream message, closed 1m candle (x = true)
    {
        "symbol": "BTCUSDT", "interval": "1m", "source": "websocket", "ingested_at": NOW,
        "payload": {"stream": "btcusdt@kline_1m", "data": {
            "e": "kline", "E": 1760000460005, "s": "BTCUSDT",
            "k": {"t": 1760000400000, "T": 1760000459999, "s": "BTCUSDT", "i": "1m",
                  "o": "121300.00", "c": "121320.40", "h": "121335.00", "l": "121295.10",
                  "v": "6.512", "n": 1023, "x": True, "q": "790010.33",
                  "V": "3.1", "Q": "376100.20", "B": "0"}}},
    },
    # WebSocket: candle still open (x = false) -> transform.py must skip it
    {
        "symbol": "BTCUSDT", "interval": "1m", "source": "websocket", "ingested_at": NOW,
        "payload": {"stream": "btcusdt@kline_1m", "data": {
            "e": "kline", "E": 1760000470005, "s": "BTCUSDT",
            "k": {"t": 1760000460000, "T": 1760000519999, "s": "BTCUSDT", "i": "1m",
                  "o": "121320.40", "c": "121310.00", "h": "121325.00", "l": "121305.00",
                  "v": "1.200", "n": 210, "x": False, "q": "145570.10",
                  "V": "0.6", "Q": "72790.00", "B": "0"}}},
    },
]

if __name__ == "__main__":
    col = get_mongo_collection()
    col.insert_many(SAMPLE_DOCS)
    print(f"Inserted {len(SAMPLE_DOCS)} sample documents, raw_klines now has {col.count_documents({})}")
