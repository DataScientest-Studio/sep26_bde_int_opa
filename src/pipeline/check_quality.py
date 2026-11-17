"""Data quality gate for the batch pipeline (last task of the Airflow DAG).

Owner: Talgat

Checks, per symbol (interval 15m by default):
1. Freshness: the newest candle in fact_kline is at most MAX_AGE_MINUTES old.
2. Gold coverage: every candle in fact_kline has a row in gold_features.
3. Outcomes: the longest-horizon outcomes (4 candles) reach the newest candle minus 4 candles.

Prints one line per symbol and exits with code 1 if any check fails,
so Airflow marks the task (and the run) as failed.

Run from the repo root:
    python src/pipeline/check_quality.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

INTERVAL = os.getenv("REST_INTERVAL", "15m")
MAX_AGE_MINUTES = int(os.getenv("QUALITY_MAX_AGE_MINUTES", "45"))
MAX_HORIZON_CANDLES = 4
CANDLE_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

STATS_SQL = """
SELECT s.symbol,
       (SELECT MAX(open_time) FROM fact_kline f WHERE f.symbol_id = s.symbol_id AND f.interval = %(interval)s) AS last_candle,
       (SELECT COUNT(*) FROM fact_kline f WHERE f.symbol_id = s.symbol_id AND f.interval = %(interval)s)       AS candles,
       (SELECT COUNT(*) FROM gold_features g WHERE g.symbol_id = s.symbol_id AND g.interval = %(interval)s)    AS feature_rows,
       (SELECT MAX(open_time) FROM outcomes o WHERE o.symbol_id = s.symbol_id AND o.interval = %(interval)s
           AND o.horizon_candles = %(max_horizon)s)                                                       AS last_outcome
FROM dim_symbol s
ORDER BY s.symbol;
"""


def evaluate(row, now, max_age=timedelta(minutes=MAX_AGE_MINUTES), interval=INTERVAL):
    """Pure check of one symbol's stats -> list of problems (empty = OK)."""
    symbol, last_candle, candles, feature_rows, last_outcome = row
    if not candles:
        return [f"{symbol}: no candles in fact_kline"]
    last_candle = last_candle.astimezone(timezone.utc)
    problems = []
    age = now - last_candle
    if age > max_age:
        minutes = int(age.total_seconds() // 60)
        problems.append(f"{symbol}: newest candle {last_candle:%Y-%m-%d %H:%M} UTC is {minutes} min old")
    if feature_rows != candles:
        problems.append(f"{symbol}: gold_features has {feature_rows} rows, fact_kline has {candles}")
    step = timedelta(minutes=CANDLE_MINUTES.get(interval, 15))
    if last_outcome is None or last_outcome < last_candle - MAX_HORIZON_CANDLES * step:
        last = last_outcome.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if last_outcome else "none"
        problems.append(f"{symbol}: outcomes lag behind (last {last})")
    return problems


def main():
    from db import get_pg_connection

    now = datetime.now(timezone.utc)
    with get_pg_connection() as conn, conn.cursor() as cur:
        cur.execute(STATS_SQL, {"interval": INTERVAL, "max_horizon": MAX_HORIZON_CANDLES})
        rows = cur.fetchall()

    all_problems = []
    for row in rows:
        problems = evaluate(row, now)
        all_problems += problems
        status = "OK  " if not problems else "FAIL"
        newest = row[1].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if row[1] else "-"
        print(f"{status} {row[0]} {INTERVAL}: {row[2]} candles, {row[3]} feature rows, newest {newest}")
    for p in all_problems:
        print("  -", p)
    if all_problems:
        sys.exit(1)
    print("Quality checks passed.")


if __name__ == "__main__":
    main()
