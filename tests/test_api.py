"""Unit tests for the FastAPI app (src/api/main.py) with a fake database - no Postgres needed."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src.api` is importable
from src.api import main  # noqa: E402

T0 = datetime(2026, 10, 24, 15, 45, tzinfo=timezone.utc)


def candle(i):
    t = T0 - timedelta(minutes=15 * i)
    return {"symbol": "BTCUSDT", "interval": "15m", "open_time": t, "close_time": t + timedelta(minutes=15),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
            "quote_volume": 1005.0, "trades": 42}


class FakeDB:
    """Returns canned rows per query name; records the calls."""

    def __init__(self, rows=None, ping_ok=True):
        self.rows = rows or {}
        self.ping_ok = ping_ok
        self.calls = []

    def fetch_all(self, query_name, params=None):
        self.calls.append((query_name, params))
        return self.rows.get(query_name, [])

    def ping(self):
        if not self.ping_ok:
            raise ConnectionError("down")
        return True


def client_with(db):
    main.app.dependency_overrides[main.get_db] = lambda: db
    return TestClient(main.app)


def teardown_function():
    main.app.dependency_overrides.clear()


def test_every_query_file_exists():
    for name in ("symbols", "candles", "stats", "latest_features", "latest_prediction", "baseline_direction"):
        assert main.load_query(name).strip()


def test_health():
    assert client_with(FakeDB()).get("/health").json() == {"status": "ok"}


def test_ready_ok_and_db_down():
    assert client_with(FakeDB()).get("/ready").status_code == 200
    assert client_with(FakeDB(ping_ok=False)).get("/ready").status_code == 503


def test_candles_oldest_first_and_params_passed():
    db = FakeDB({"candles": [candle(0), candle(1), candle(2)]})   # SQL returns newest first
    r = client_with(db).get("/candles", params={"symbol": "BTCUSDT", "limit": 3})
    assert r.status_code == 200
    times = [c["open_time"] for c in r.json()]
    assert times == sorted(times)
    assert db.calls[0] == ("candles", {"symbol": "BTCUSDT", "interval": "15m", "limit": 3})


def test_candles_validation_and_404():
    c = client_with(FakeDB())
    assert c.get("/candles", params={"limit": 5000}).status_code == 422          # above max
    assert c.get("/candles", params={"symbol": "btc;drop"}).status_code == 422   # bad symbol
    assert c.get("/candles", params={"symbol": "ETHUSDT"}).status_code == 404    # no data


def test_stats_symbols_features():
    stat = {"symbol": "BTCUSDT", "interval": "15m", "candles": 17279, "first_candle": T0, "last_candle": T0,
            "min_price": 60000.0, "max_price": 126000.0, "avg_close": 95000.5, "total_volume": 1.5e6}
    db = FakeDB({"stats": [stat], "symbols": [{"symbol": "BTCUSDT", "base_asset": "BTC", "quote_asset": "USDT"}],
                 "latest_features": [{"symbol": "BTCUSDT", "open_time": T0, "rsi_14": 55.0, "rv_16": 0.002}]})
    c = client_with(db)
    r = c.get("/stats")
    assert r.status_code == 200 and r.json()[0]["candles"] == 17279
    assert c.get("/symbols").json()[0]["symbol"] == "BTCUSDT"
    assert c.get("/features/latest").json()["rsi_14"] == 55.0
    assert client_with(FakeDB()).get("/features/latest").status_code == 404


def test_predict_uses_active_model_when_present():
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": 0.002}], "latest_prediction": [{
        "symbol": "BTCUSDT", "interval": "15m", "task": "direction", "open_time": T0, "horizon_candles": 1,
        "y_pred_label": 1, "y_pred_value": 0.57, "y_baseline": 0.5, "model_version": "v1",
        "predicted_at": T0}]})
    c = client_with(db)
    body = c.get("/predict/BTCUSDT").json()
    assert body["source"] == "model" and body["model_version"] == "v1" and body["stale"] is False
    assert body["prob_up"] == 0.57 and body["predicted_up"] is True
    assert c.get("/predict", params={"symbol": "BTCUSDT"}).json() == body     # query-parameter route


def test_predict_flags_stale_model_prediction():
    old = T0 - timedelta(hours=6)
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": 0.002}], "latest_prediction": [{
        "open_time": old, "horizon_candles": 1, "y_pred_label": None, "y_pred_value": 0.51,
        "model_version": "v1"}]})
    body = client_with(db).get("/predict/BTCUSDT").json()
    assert body["stale"] is True and body["predicted_up"] is None


def test_predict_direction_falls_back_to_baseline():
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": 0.002}],
                 "baseline_direction": [{"prob_up": 0.4375, "n": 96}]})
    body = client_with(db).get("/predict/SOLUSDT", params={"task": "direction"}).json()
    assert body["source"] == "baseline" and body["horizon_candles"] == 1
    assert body["prob_up"] == 0.4375 and body["predicted_up"] is False


def test_predict_direction_baseline_without_outcomes():
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": 0.002}],
                 "baseline_direction": [{"prob_up": None, "n": 0}]})
    body = client_with(db).get("/predict/BTCUSDT").json()
    assert body["prob_up"] is None and body["predicted_up"] is None


def test_predict_volatility_warmup_returns_503():
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": None}]})
    assert client_with(db).get("/predict/BTCUSDT", params={"task": "volatility"}).status_code == 503


def test_predict_volatility_baseline_and_unknown_task():
    db = FakeDB({"latest_features": [{"open_time": T0, "rv_16": 0.002}]})
    c = client_with(db)
    body = c.get("/predict/BTCUSDT", params={"task": "volatility"}).json()
    assert body["predicted_vol"] == 0.002 and body["horizon_candles"] == 4 and body["source"] == "baseline"
    assert c.get("/predict/BTCUSDT", params={"task": "moon"}).status_code == 422


def test_predict_404_without_features():
    assert client_with(FakeDB()).get("/predict/BTCUSDT").status_code == 404
