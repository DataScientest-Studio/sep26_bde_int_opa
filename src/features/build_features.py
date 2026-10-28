"""GOLD: fact_kline (silver) -> gold_features + outcomes.

Owner: Talgat  (Stage 2)

What it does:
1. Reads closed candles per symbol/interval from fact_kline
2. compute_features(): 34 features per candle, each using only candles with
   open_time <= the row's open_time (no look-ahead). Pure pandas, no DB.
3. compute_outcomes(): what really happened after candle t for each horizon
   (1 candle, 4 candles). These are the training targets and the truth that
   predictions are later scored against.
4. Upserts both into Postgres, so running it again never creates duplicates.

Feature set 'v1' is the contract with the ML side: the column list below
(FEATURE_COLS) must match gold_features in sql/init.sql.

Run from the repo root:
    python src/features/build_features.py                  # all symbols, 15m
    python src/features/build_features.py --interval 1m
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from db import get_pg_connection  # noqa: E402

FEATURE_SET = "v1"
HORIZONS = (1, 4)            # candles ahead: next candle, next hour (15m candles)
LARGE_MOVE_THR = 0.01        # |log return| > 1 % counts as a large move

FEATURE_COLS = [
    "ret_1", "ret_4", "ret_16", "ret_96", "ret_1_lag1", "ret_1_lag2", "ret_1_lag3",
    "sma_20_ratio", "sma_96_ratio", "ema_12_ratio", "ema_26_ratio", "macd_rel", "rsi_14",
    "rv_16", "rv_96", "gk_var_16", "atr_14_rel",
    "hl_range_rel", "body_rel", "upper_wick_rel", "lower_wick_rel",
    "vol_z_96", "vol_ratio_16", "trades_z_96", "log_quote_volume",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "min_to_funding",
    "btc_ret_1", "btc_ret_16",
]

READ_SQL = """
SELECT s.symbol, f.open_time, f.open, f.high, f.low, f.close, f.volume, f.quote_volume, f.trades
FROM fact_kline f JOIN dim_symbol s USING (symbol_id)
WHERE f.interval = %(interval)s
ORDER BY s.symbol, f.open_time;
"""

UPSERT_FEATURES_SQL = """
INSERT INTO gold_features (symbol_id, interval, open_time, feature_set, {cols}) VALUES %s
ON CONFLICT (symbol_id, interval, open_time) DO UPDATE SET
    feature_set = EXCLUDED.feature_set, {updates}, computed_at = NOW();
""".format(cols=", ".join(FEATURE_COLS), updates=", ".join(f"{c} = EXCLUDED.{c}" for c in FEATURE_COLS))

UPSERT_OUTCOMES_SQL = """
INSERT INTO outcomes (symbol_id, interval, open_time, horizon_candles,
                      realized_ret, realized_dir, realized_vol, realized_large, large_move_thr) VALUES %s
ON CONFLICT (symbol_id, interval, open_time, horizon_candles) DO UPDATE SET
    realized_ret = EXCLUDED.realized_ret, realized_dir = EXCLUDED.realized_dir,
    realized_vol = EXCLUDED.realized_vol, realized_large = EXCLUDED.realized_large,
    large_move_thr = EXCLUDED.large_move_thr, filled_at = NOW();
"""


# ---------------------------------------------------------------- pure functions

def rsi(close, window=14):
    """Classic RSI with Wilder's smoothing (0..100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df, btc=None):
    """One symbol: DataFrame sorted by open_time with columns
    open, high, low, close, volume, quote_volume, trades  ->  same index + FEATURE_COLS.

    `btc` is the BTCUSDT frame on the same grid (for cross-asset features);
    None means the frame itself is BTC.
    Every value at row t uses only rows <= t.
    """
    df = df.sort_values("open_time").reset_index(drop=True)
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    v, q, n = df["volume"].astype(float), df["quote_volume"].astype(float), df["trades"].astype(float)
    out = pd.DataFrame({"open_time": df["open_time"]})

    log_c = np.log(c)
    ret_1 = log_c.diff()
    out["ret_1"] = ret_1
    out["ret_4"] = log_c.diff(4)
    out["ret_16"] = log_c.diff(16)
    out["ret_96"] = log_c.diff(96)
    out["ret_1_lag1"] = ret_1.shift(1)
    out["ret_1_lag2"] = ret_1.shift(2)
    out["ret_1_lag3"] = ret_1.shift(3)

    out["sma_20_ratio"] = c / c.rolling(20).mean() - 1
    out["sma_96_ratio"] = c / c.rolling(96).mean() - 1
    ema_12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["ema_12_ratio"] = c / ema_12 - 1
    out["ema_26_ratio"] = c / ema_26 - 1
    out["macd_rel"] = (ema_12 - ema_26) / c
    out["rsi_14"] = rsi(c, 14)

    out["rv_16"] = ret_1.rolling(16).std()
    out["rv_96"] = ret_1.rolling(96).std()
    # Garman-Klass variance per candle, averaged over 16 candles
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    out["gk_var_16"] = gk.rolling(16).mean()
    prev_c = c.shift(1)
    true_range = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14_rel"] = true_range.rolling(14).mean() / c

    rng = (h - l).replace(0, np.nan)
    out["hl_range_rel"] = (h - l) / c
    out["body_rel"] = (c - o) / rng
    out["upper_wick_rel"] = (h - np.maximum(o, c)) / rng
    out["lower_wick_rel"] = (np.minimum(o, c) - l) / rng

    out["vol_z_96"] = (v - v.rolling(96).mean()) / v.rolling(96).std()
    out["vol_ratio_16"] = v / v.rolling(16).mean()
    out["trades_z_96"] = (n - n.rolling(96).mean()) / n.rolling(96).std()
    out["log_quote_volume"] = np.log1p(q)

    t = pd.to_datetime(df["open_time"], utc=True)
    hour = t.dt.hour + t.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    minutes = t.dt.hour * 60 + t.dt.minute
    out["min_to_funding"] = (480 - minutes % 480) % 480   # funding at 00:00 / 08:00 / 16:00 UTC

    if btc is None:
        out["btc_ret_1"] = out["ret_1"]
        out["btc_ret_16"] = out["ret_16"]
    else:
        b = btc.sort_values("open_time").set_index("open_time")["close"].astype(float)
        b_log = np.log(b)
        out["btc_ret_1"] = out["open_time"].map(b_log.diff()).values
        out["btc_ret_16"] = out["open_time"].map(b_log.diff(16)).values

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[["open_time"] + FEATURE_COLS]


def compute_outcomes(df, horizons=HORIZONS, thr=LARGE_MOVE_THR):
    """One symbol -> long frame: open_time, horizon_candles, realized_ret, realized_dir,
    realized_vol, realized_large. Rows whose future is not in df yet are dropped.
    """
    df = df.sort_values("open_time").reset_index(drop=True)
    log_c = np.log(df["close"].astype(float))
    ret_1 = log_c.diff()
    frames = []
    for hz in horizons:
        realized_ret = log_c.shift(-hz) - log_c
        if hz > 1:
            # std of the next hz one-candle returns (t+1 .. t+hz)
            realized_vol = ret_1[::-1].rolling(hz).std()[::-1].shift(-1)
        else:
            realized_vol = pd.Series(np.nan, index=df.index)
        f = pd.DataFrame({
            "open_time": df["open_time"],
            "horizon_candles": hz,
            "realized_ret": realized_ret,
            "realized_dir": (realized_ret > 0).astype(int),
            "realized_vol": realized_vol,
            "realized_large": realized_ret.abs() > thr,
        }).dropna(subset=["realized_ret"])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- I/O

def _nan_to_none(values):
    return [None if (isinstance(x, float) and np.isnan(x)) else x for x in values]


def run(interval="15m"):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(READ_SQL, {"interval": interval})
            raw = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
            cur.execute("SELECT symbol, symbol_id FROM dim_symbol;")
            symbol_ids = dict(cur.fetchall())

        if raw.empty:
            print(f"No candles for interval {interval} in fact_kline")
            return

        btc = raw[raw["symbol"] == "BTCUSDT"]
        for symbol, grp in raw.groupby("symbol"):
            sid = symbol_ids[symbol]
            feats = compute_features(grp, btc=None if symbol == "BTCUSDT" else btc)
            outs = compute_outcomes(grp)

            feat_rows = [
                (sid, interval, r.open_time, FEATURE_SET, *_nan_to_none([float(getattr(r, col)) for col in FEATURE_COLS]))
                for r in feats.itertuples(index=False)
            ]
            out_rows = [
                (sid, interval, r.open_time, int(r.horizon_candles), float(r.realized_ret), int(r.realized_dir),
                 None if np.isnan(r.realized_vol) else float(r.realized_vol), bool(r.realized_large), LARGE_MOVE_THR)
                for r in outs.itertuples(index=False)
            ]
            with conn.cursor() as cur:
                execute_values(cur, UPSERT_FEATURES_SQL, feat_rows, page_size=2000)
                execute_values(cur, UPSERT_OUTCOMES_SQL, out_rows, page_size=2000)
            conn.commit()
            print(f"{symbol} {interval}: {len(feat_rows)} feature rows, {len(out_rows)} outcome rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="15m")
    run(parser.parse_args().interval)
