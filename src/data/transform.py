"""SILVER: MongoDB raw_klines (bronze) -> PostgreSQL fact_kline (silver).

Owner: Talgat

What it does:
1. Reads raw documents from Mongo `raw_klines`
2. Parses the payload (REST = array, WebSocket = object with "k")
3. Cleans it: strings -> numbers, milliseconds -> UTC timestamps, named columns
4. Skips candles that are not closed yet (they would change later)
5. Upserts into Postgres `fact_kline`, so running it again never creates duplicates

Run from the repo root:
    python src/data/transform.py
"""
from datetime import datetime, timezone
from decimal import Decimal

from psycopg2.extras import execute_values

from db import get_mongo_collection, get_pg_connection

UPSERT_SQL = """
INSERT INTO fact_kline (
    symbol_id, interval, open_time, close_time,
    open, high, low, close, volume, quote_volume, trades, source
) VALUES %s
ON CONFLICT (symbol_id, interval, open_time) DO UPDATE SET
    close_time   = EXCLUDED.close_time,
    open         = EXCLUDED.open,
    high         = EXCLUDED.high,
    low          = EXCLUDED.low,
    close        = EXCLUDED.close,
    volume       = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume,
    trades       = EXCLUDED.trades,
    source       = EXCLUDED.source,
    loaded_at    = NOW();
"""


def ms_to_utc(ms):
    """Binance timestamps are milliseconds since 1970 -> timezone-aware datetime."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def parse_rest(payload):
    """REST kline is an array:
    [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
    """
    return {
        "open_time": ms_to_utc(payload[0]),
        "open": Decimal(payload[1]),
        "high": Decimal(payload[2]),
        "low": Decimal(payload[3]),
        "close": Decimal(payload[4]),
        "volume": Decimal(payload[5]),
        "close_time": ms_to_utc(payload[6]),
        "quote_volume": Decimal(payload[7]),
        "trades": int(payload[8]),
        "is_closed": None,  # REST does not say; decided by close_time below
    }


def find_ws_kline(payload):
    """WebSocket payload can be stored as the full message, as "data" or as "k" itself."""
    if "data" in payload:          # combined stream: {"stream": ..., "data": {..., "k": {...}}}
        payload = payload["data"]
    if "k" in payload:             # single stream message: {"e": "kline", ..., "k": {...}}
        payload = payload["k"]
    return payload


def parse_ws(payload):
    """WebSocket kline object (named fields):
    t=open_time, T=close_time, o/h/l/c=prices, v=volume, q=quote_volume, n=trades, x=is_closed
    """
    k = find_ws_kline(payload)
    return {
        "open_time": ms_to_utc(k["t"]),
        "close_time": ms_to_utc(k["T"]),
        "open": Decimal(k["o"]),
        "high": Decimal(k["h"]),
        "low": Decimal(k["l"]),
        "close": Decimal(k["c"]),
        "volume": Decimal(k["v"]),
        "quote_volume": Decimal(k["q"]),
        "trades": int(k["n"]),
        "is_closed": bool(k["x"]),
    }


def parse_document(doc, now=None):
    """Turn one bronze document into one clean row, or None if it should be skipped."""
    now = now or datetime.now(timezone.utc)
    if doc["source"] == "rest":
        row = parse_rest(doc["payload"])
    elif doc["source"] == "websocket":
        row = parse_ws(doc["payload"])
    else:
        raise ValueError(f"Unknown source: {doc['source']}")

    # Skip candles that are still open: their prices will still change.
    if row["is_closed"] is False or row["close_time"] > now:
        return None

    row["symbol"] = doc["symbol"]
    row["interval"] = doc["interval"]
    row["source"] = doc["source"]
    return row


def load_symbol_ids(cur):
    cur.execute("SELECT symbol, symbol_id FROM dim_symbol;")
    return dict(cur.fetchall())


def run():
    col = get_mongo_collection()
    docs = list(col.find({}, {"_id": 0}))
    print(f"Read {len(docs)} documents from Mongo raw_klines")

    rows, skipped = [], 0
    for doc in docs:
        row = parse_document(doc)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    # The same candle can come from REST and WebSocket: keep one per key (last one wins).
    unique = {(r["symbol"], r["interval"], r["open_time"]): r for r in rows}

    with get_pg_connection() as conn, conn.cursor() as cur:
        symbol_ids = load_symbol_ids(cur)
        unknown = {r["symbol"] for r in unique.values()} - symbol_ids.keys()
        if unknown:
            raise ValueError(f"Symbols missing in dim_symbol: {unknown}")

        values = [
            (
                symbol_ids[r["symbol"]], r["interval"], r["open_time"], r["close_time"],
                r["open"], r["high"], r["low"], r["close"],
                r["volume"], r["quote_volume"], r["trades"], r["source"],
            )
            for r in unique.values()
        ]
        if values:
            execute_values(cur, UPSERT_SQL, values, page_size=1000)

        cur.execute("SELECT COUNT(*) FROM fact_kline;")
        total = cur.fetchone()[0]

    print(f"Upserted {len(values)} candles, skipped {skipped} open candles")
    print(f"fact_kline now has {total} rows")


if __name__ == "__main__":
    run()
