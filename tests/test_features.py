"""Unit tests for src/features/build_features.py (pure functions, no database)."""
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "features"))
for mod in ("psycopg2", "psycopg2.extras", "pymongo", "dotenv"):
    sys.modules.setdefault(mod, MagicMock())

import build_features as bf  # noqa: E402


def make_candles(n=300, seed=0, start="2026-10-01 00:00+00"):
    """Synthetic 15m random-walk candles with the fact_kline columns."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    return pd.DataFrame({
        "open_time": pd.date_range(start, periods=n, freq="15min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(10, 100, n), "quote_volume": rng.uniform(1e5, 1e6, n),
        "trades": rng.integers(100, 1000, n),
    })


def test_feature_columns_match_sql_schema():
    sql = (ROOT / "sql" / "init.sql").read_text()
    gold = sql[sql.index("CREATE TABLE IF NOT EXISTS gold_features"):sql.index("CREATE TABLE IF NOT EXISTS outcomes")]
    sql_cols = re.findall(r"^\s+([a-z_0-9]+)\s+DOUBLE PRECISION", gold, flags=re.M)
    assert sql_cols == bf.FEATURE_COLS


def test_output_has_all_features_and_no_nan_after_warmup():
    feats = bf.compute_features(make_candles())
    assert list(feats.columns) == ["open_time"] + bf.FEATURE_COLS
    assert feats.iloc[100:][bf.FEATURE_COLS].isna().sum().sum() == 0   # longest window is 96


def test_no_lookahead():
    """Feature at row t must not change when later candles are removed (point-in-time rule)."""
    df = make_candles()
    full = bf.compute_features(df)
    for t in (120, 200, 299):
        truncated = bf.compute_features(df.iloc[:t + 1])
        a, b = full.iloc[t][bf.FEATURE_COLS].astype(float), truncated.iloc[-1][bf.FEATURE_COLS].astype(float)
        assert np.allclose(a, b, equal_nan=True), f"look-ahead at row {t}"


def test_value_ranges():
    feats = bf.compute_features(make_candles()).iloc[100:]
    assert feats["rsi_14"].between(0, 100).all()
    assert feats["body_rel"].abs().le(1).all()
    assert feats["upper_wick_rel"].between(0, 1).all() and feats["lower_wick_rel"].between(0, 1).all()
    assert feats["min_to_funding"].between(0, 479).all()
    assert np.allclose(feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2, 1)


def test_cross_asset_uses_btc_on_same_timestamp():
    btc, sol = make_candles(seed=1), make_candles(seed=2)
    f_sol = bf.compute_features(sol, btc=btc)
    f_btc = bf.compute_features(btc)
    assert np.allclose(f_sol["btc_ret_1"].iloc[5:], f_btc["ret_1"].iloc[5:])
    assert np.allclose(f_btc["btc_ret_1"].iloc[5:], f_btc["ret_1"].iloc[5:])   # BTC itself


def test_outcomes_match_future_closes():
    df = make_candles(n=50)
    outs = bf.compute_outcomes(df, horizons=(1, 4), thr=0.01)
    log_c = np.log(df["close"].values)
    h1 = outs[outs.horizon_candles == 1].reset_index(drop=True)
    h4 = outs[outs.horizon_candles == 4].reset_index(drop=True)
    assert len(h1) == 49 and len(h4) == 46                      # last h rows have no future yet
    assert np.isclose(h1.loc[10, "realized_ret"], log_c[11] - log_c[10])
    assert np.isclose(h4.loc[10, "realized_ret"], log_c[14] - log_c[10])
    assert h1.loc[10, "realized_dir"] == int(log_c[11] > log_c[10])
    expected_vol = np.std(np.diff(log_c[10:15]), ddof=1)       # returns t+1..t+4
    assert np.isclose(h4.loc[10, "realized_vol"], expected_vol)
    assert h1["realized_vol"].isna().all()
    assert (h4["realized_large"] == (h4["realized_ret"].abs() > 0.01)).all()
