"""BRONZE: historical klines from Binance REST -> MongoDB (raw_klines).

Owner: Dennis

Downloads finished candles for every symbol in SYMBOLS (default: BTCUSDT and
SOLUSDT, REST_INTERVAL 15m, the last HISTORY_DAYS = 180 days) and stores each
kline exactly as Binance sent it: one document per kline, in the format agreed
in the README.

Run it as often as you like: it continues after the newest candle already in
MongoDB, so the same candle is never stored twice.

Run from the repo root, with the virtual environment active:
    python src/data/extract_rest.py
"""
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from db import get_mongo_collection

load_dotenv()

# --- Settings (read from .env; the defaults match .env.example) -------------
BINANCE_REST_URL = os.getenv("BINANCE_REST_URL", "https://api.binance.com")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,SOLUSDT").split(",") if s.strip()]
INTERVAL = os.getenv("REST_INTERVAL", "15m")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "180"))
LIMIT = 1000  # the most klines Binance returns per request

# How long one candle lasts, in milliseconds
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
DAY_MS = 86_400_000


# --- Small helpers ----------------------------------------------------------
def to_text(ms):
    """Turn a Binance timestamp (milliseconds since 1970, UTC) into readable text."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def last_closed_end(now_ms, step_ms):
    """Last millisecond before the candle that is still running.

    Asking Binance for candles up to this moment means we only ever receive
    finished candles, never the one whose prices are still changing.
    """
    current_candle_open = now_ms - now_ms % step_ms
    return current_candle_open - 1


def make_documents(symbol, interval, klines):
    """One document per kline, in the agreed raw_klines format (see README)."""
    ingested_at = datetime.now(timezone.utc)
    return [
        {
            "symbol": symbol,
            "interval": interval,
            "source": "rest",
            "ingested_at": ingested_at,
            "payload": kline,  # the kline exactly as Binance sent it (array of 12 values)
        }
        for kline in klines
    ]


# --- MongoDB ----------------------------------------------------------------
def check_mongo(collection):
    """Stop early with a clear message if MongoDB is down or the login is wrong."""
    try:
        collection.database.list_collection_names()
    except ServerSelectionTimeoutError:
        raise SystemExit("Cannot reach MongoDB. Is it running? Start it with: docker compose up -d\n"
                         "If it is running, check MONGO_HOST and MONGO_PORT in your .env file.")
    except OperationFailure:
        raise SystemExit("MongoDB refused the login. Check MONGO_USER and MONGO_PASSWORD in .env.")


def newest_saved_open_time(collection, symbol, interval):
    """Open time (ms) of the newest REST candle already stored for this symbol, or None."""
    result = list(collection.aggregate([
        {"$match": {"source": "rest", "symbol": symbol, "interval": interval}},
        # payload[0] is the kline's open time: keep the largest one
        {"$group": {"_id": None, "newest": {"$max": {"$arrayElemAt": ["$payload", 0]}}}},
    ]))
    return result[0]["newest"] if result else None


# --- Binance ----------------------------------------------------------------
def get_klines(symbol, interval, start_ms, end_ms):
    """Ask Binance for up to 1000 klines between start_ms and end_ms."""
    params = {"symbol": symbol, "interval": interval,
              "startTime": start_ms, "endTime": end_ms, "limit": LIMIT}
    for attempt in range(1, 6):
        try:
            response = requests.get(f"{BINANCE_REST_URL}/api/v3/klines", params=params, timeout=10)
        except requests.RequestException as error:
            print(f"  network problem ({error}); retry {attempt}/5 in 5 s")
            time.sleep(5)
            continue
        if response.status_code in (418, 429):  # we sent too many requests
            wait = int(response.headers.get("Retry-After", 60))
            print(f"  Binance asks us to slow down; waiting {wait} s")
            time.sleep(wait)
            continue
        if response.status_code == 451:
            raise SystemExit("Binance blocks requests from your location (HTTP 451). Turn off any VPN.")
        if not response.ok:
            raise SystemExit(f"Binance error {response.status_code}: {response.text}")
        return response.json()
    raise SystemExit("Binance did not answer after 5 tries. Check your internet connection.")


# --- The extraction ---------------------------------------------------------
def extract(collection, symbol, interval):
    """Download every new finished candle for one symbol and store it."""
    step = INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    end_ms = last_closed_end(now_ms, step)
    window_start = now_ms - HISTORY_DAYS * DAY_MS

    newest = newest_saved_open_time(collection, symbol, interval)
    if newest is None:
        start_ms = window_start
        print(f"{symbol} {interval}: first run, fetching the last {HISTORY_DAYS} days")
    else:
        # Continue after the newest stored candle, but never reach back further than HISTORY_DAYS.
        start_ms = max(newest + step, window_start)
        print(f"{symbol} {interval}: continuing after {to_text(newest)}")

    calls = candles = 0
    while start_ms <= end_ms:
        klines = get_klines(symbol, interval, start_ms, end_ms)
        if not klines:
            break
        collection.insert_many(make_documents(symbol, interval, klines))
        calls += 1
        candles += len(klines)
        print(f"  saved {len(klines)} candles, up to {to_text(klines[-1][0])}")
        start_ms = klines[-1][0] + step  # the next request starts with the next candle
        time.sleep(0.2)                  # be polite to Binance
    print(f"{symbol} {interval}: {candles} new candles from {calls} requests")


def main():
    if INTERVAL not in INTERVAL_MS:
        raise SystemExit(f"REST_INTERVAL={INTERVAL} is not supported. Use one of: {', '.join(INTERVAL_MS)}")
    collection = get_mongo_collection()
    check_mongo(collection)
    for symbol in SYMBOLS:
        extract(collection, symbol, INTERVAL)
    print("Done.")


if __name__ == "__main__":
    main()
