"""BRONZE check: what is in raw_klines, and two sample documents for the report.

Owner: Dennis

Prints how many candles raw_klines holds per source, symbol and interval, the
time range they cover, and whether any REST candles are missing or doubled.
Then writes one example document per source to docs/samples/ (the examples of
collected data for the Stage 1 report).

Run from the repo root, with the virtual environment active:
    python src/data/check_bronze.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from db import get_mongo_collection

load_dotenv()

SAMPLES_DIR = Path("docs") / "samples"

# How long one candle lasts, in milliseconds
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# --- Small helpers ----------------------------------------------------------
def to_text(ms):
    """Turn a Binance timestamp (milliseconds since 1970, UTC) into readable text."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def open_time_of(doc):
    """Open time (ms) of the candle in a bronze document, for both payload shapes."""
    payload = doc.get("payload")
    if isinstance(payload, list):    # REST: [open_time, open, high, ...]
        return payload[0]
    if isinstance(payload, dict):    # WebSocket: {"stream": ..., "data": {..., "k": {"t": open_time, ...}}}
        return payload.get("data", payload).get("k", {}).get("t")
    return None


def json_default(value):
    """Write dates as ISO text (UTC) in the sample files."""
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


def check_mongo(collection):
    """Stop early with a clear message if MongoDB is down or the login is wrong."""
    try:
        collection.database.list_collection_names()
    except ServerSelectionTimeoutError:
        raise SystemExit("Cannot reach MongoDB. Is it running? Start it with: docker compose up -d")
    except OperationFailure:
        raise SystemExit("MongoDB refused the login. Check MONGO_USER and MONGO_PASSWORD in .env.")


# --- Counting ---------------------------------------------------------------
def summarise(collection):
    """Collect the open times per (source, symbol, interval)."""
    groups = {}
    for doc in collection.find({}, {"_id": 0, "source": 1, "symbol": 1, "interval": 1, "payload": 1}):
        key = (str(doc.get("source")), str(doc.get("symbol")), str(doc.get("interval")))
        open_time = open_time_of(doc)
        if open_time is not None:
            groups.setdefault(key, []).append(open_time)
    return groups


def completeness(source, interval, open_times):
    """For REST data: are all candles between the first and the last there, each once?"""
    if source != "rest" or interval not in INTERVAL_MS:
        return ""
    unique = set(open_times)
    expected = (max(unique) - min(unique)) // INTERVAL_MS[interval] + 1
    missing, doubled = expected - len(unique), len(open_times) - len(unique)
    if missing == 0 and doubled == 0:
        return "complete"
    return ", ".join(part for part in (f"{missing} missing" if missing else "",
                                       f"{doubled} doubled" if doubled else "") if part)


def print_summary(groups):
    if not groups:
        print("raw_klines is empty. Run extract_rest.py or extract_ws.py first.")
        return
    print(f"{'source':<10} {'symbol':<8} {'interval':<8} {'candles':>8}  {'check':<20} covers")
    for (source, symbol, interval), open_times in sorted(groups.items()):
        covers = f"{to_text(min(open_times))} -> {to_text(max(open_times))}"
        check = completeness(source, interval, open_times)
        print(f"{source:<10} {symbol:<8} {interval:<8} {len(open_times):>8}  {check:<20} {covers}")


# --- Samples for the report -------------------------------------------------
def export_samples(collection):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for source in ("rest", "websocket"):
        doc = collection.find_one({"source": source}, {"_id": 0}, sort=[("ingested_at", 1)])  # the oldest one
        if doc is None:
            print(f"No {source} documents yet, so no {source} sample written.")
            continue
        path = SAMPLES_DIR / f"bronze_{source}_example.json"
        path.write_text(json.dumps(doc, indent=2, default=json_default), encoding="utf-8")
        print(f"Wrote {path}")


def main():
    collection = get_mongo_collection()
    check_mongo(collection)
    print_summary(summarise(collection))
    export_samples(collection)


if __name__ == "__main__":
    main()
