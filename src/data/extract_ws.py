"""BRONZE: live klines from Binance WebSocket -> MongoDB (raw_klines).

Owner: Dennis

Listens to Binance's live candles (WS_INTERVAL, default 1m) for every symbol in
SYMBOLS and stores each finished candle (k.x = true) as one document. The
payload is the whole message exactly as Binance sent it (format: see README).

Runs until you press Ctrl+C. If the connection drops (Binance closes every
connection after 24 hours), it reconnects by itself.

Run from the repo root, with the virtual environment active:
    python src/data/extract_ws.py
"""
import json
import os
import time
from datetime import datetime, timezone

import websocket  # from the websocket-client package
from dotenv import load_dotenv
from pymongo.errors import OperationFailure, PyMongoError, ServerSelectionTimeoutError

from db import get_mongo_collection

load_dotenv()

# --- Settings (read from .env; the defaults match .env.example) -------------
BINANCE_WS_URL = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,SOLUSDT").split(",") if s.strip()]
INTERVAL = os.getenv("WS_INTERVAL", "1m")
ONLY_CLOSED = True   # store a candle only once it has finished (x = true)
SILENCE_LIMIT = 60   # seconds without any message before we treat the connection as dead
RECONNECT_WAIT = 5   # seconds to wait before connecting again


# --- Small helpers ----------------------------------------------------------
def to_text(ms):
    """Turn a Binance timestamp (milliseconds since 1970, UTC) into readable text."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def stream_url():
    """Address of Binance's combined stream for all symbols."""
    # Binance wants lower-case names, joined with "/": btcusdt@kline_1m/solusdt@kline_1m
    names = [f"{symbol.lower()}@kline_{INTERVAL}" for symbol in SYMBOLS]
    return f"{BINANCE_WS_URL}/stream?streams={'/'.join(names)}"


def make_document(message):
    """One document per finished candle, in the agreed raw_klines format (see README)."""
    event = message["data"]
    return {
        "symbol": event["s"],
        "interval": event["k"]["i"],
        "source": "websocket",
        "ingested_at": datetime.now(timezone.utc),
        "payload": message,  # the whole message exactly as received: {"stream": ..., "data": {..., "k": {...}}}
    }


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


# --- Handling one message ---------------------------------------------------
def save_if_finished(collection, raw_message):
    """Store one Binance message if it is a finished candle. Returns True if it was stored."""
    message = json.loads(raw_message)
    event = message.get("data", {})
    if event.get("e") != "kline":
        return False  # not a candle message
    if ONLY_CLOSED and not event["k"]["x"]:
        return False  # still running; Binance sends it again once it has closed
    try:
        collection.insert_one(make_document(message))
    except PyMongoError as error:
        print(f"Could not save to MongoDB ({error}); candle skipped")
        return False
    k = event["k"]
    print(f"saved {event['s']} {k['i']} candle {to_text(k['t'])}  close {k['c']}")
    return True


# --- The listening loop -----------------------------------------------------
def listen(collection):
    """Receive messages forever. Each round of the outer loop is one connection."""
    while True:
        connection = None
        try:
            connection = websocket.create_connection(stream_url(), timeout=SILENCE_LIMIT)
            print(f"Connected. A candle is saved when it finishes, so the first one comes within {INTERVAL}.")
            while True:
                message = connection.recv()  # waits for the next message; answers Binance's pings by itself
                if not message:              # an empty message means Binance closed the connection
                    break
                save_if_finished(collection, message)
        except (websocket.WebSocketException, OSError) as error:
            print(f"Connection problem: {error}")
        finally:
            if connection is not None:
                connection.shutdown()
        print(f"Reconnecting in {RECONNECT_WAIT} s...")
        time.sleep(RECONNECT_WAIT)


def main():
    collection = get_mongo_collection()
    check_mongo(collection)
    print(f"Listening to {', '.join(SYMBOLS)} ({INTERVAL} candles). Stop with Ctrl+C.")
    try:
        listen(collection)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
