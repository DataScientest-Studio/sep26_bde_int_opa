"""Unit tests for src/models/ (training and predict), no database needed."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))

import predict_model as pm  # noqa: E402
import train_model as tm  # noqa: E402

CANDLE = pd.Timedelta(minutes=15)


def make_dataset(n=3000, symbols=("BTCUSDT", "SOLUSDT"), seed=0, signal=1.5):
    """Training-shaped rows: every feature column + realized_dir.

    With signal > 0, the target leans on ret_1, so a working model must beat a coin flip.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-05-01", periods=n, freq="15min", tz="UTC")
    frames = []
    for symbol in symbols:
        f = pd.DataFrame(rng.normal(size=(n, len(tm.FEATURE_COLS))), columns=tm.FEATURE_COLS)
        p_up = 1 / (1 + np.exp(-signal * f["ret_1"]))
        f[tm.TARGET] = (rng.uniform(size=n) < p_up).astype(int)
        f.insert(0, "open_time", times)
        f.insert(0, "symbol", symbol)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# --- Split ------------------------------------------------------------------
def test_cutoffs_are_the_same_dates_for_every_symbol():
    data = make_dataset()
    val_start, test_start = tm.time_cutoffs(data)
    assert val_start < test_start
    for symbol in ("BTCUSDT", "SOLUSDT"):
        times = data.loc[data["symbol"] == symbol, "open_time"]
        assert (times < val_start).mean() == pytest.approx(0.70, abs=0.01)
        assert (times >= test_start).mean() == pytest.approx(0.15, abs=0.01)


def test_split_is_chronological_and_purged():
    data = make_dataset(symbols=("BTCUSDT",))
    val_start, test_start = tm.time_cutoffs(data)
    train, val, test = tm.split_by_time(data, val_start, test_start, horizon=1)
    # no training label may come from the validation period, no validation label from the test period
    assert (train["open_time"] + CANDLE < val_start).all()
    assert (val["open_time"] + CANDLE < test_start).all()
    assert train["open_time"].max() < val["open_time"].min() <= val["open_time"].max() < test["open_time"].min()
    # exactly one candle is purged before each cut
    assert len(train) + len(val) + len(test) == len(data) - 2


def test_longer_horizon_purges_more():
    data = make_dataset(symbols=("BTCUSDT",))
    val_start, test_start = tm.time_cutoffs(data)
    short = tm.split_by_time(data, val_start, test_start, horizon=1)
    long = tm.split_by_time(data, val_start, test_start, horizon=4)
    assert len(short[0]) - len(long[0]) == 3


# --- Models and metrics -----------------------------------------------------
def test_evaluate_perfect_and_coin_flip():
    y = np.array([0, 1, 0, 1, 1, 0])
    perfect = tm.evaluate(y, y.astype(float))
    assert perfect["auc"] == 1.0 and perfect["accuracy"] == 1.0
    flip = tm.evaluate(y, np.full(len(y), 0.5))
    assert flip["auc"] == 0.5
    assert flip["log_loss"] == pytest.approx(np.log(2), abs=1e-6)


def test_models_find_a_real_signal_and_baseline_does_not():
    data = make_dataset(symbols=("BTCUSDT",))
    train, val, _ = tm.split_by_time(data, *tm.time_cutoffs(data))
    comparison, _ = tm.compare_models(train, val)
    assert comparison.loc["baseline", "val_auc"] == pytest.approx(0.5)
    assert comparison.loc["logreg", "val_auc"] > 0.65
    assert tm.pick_best(comparison) in ("logreg", "hgb")


def test_build_never_scores_on_training_rows():
    data = make_dataset()
    val_start, test_start = tm.time_cutoffs(data)
    built = tm.build(data, "SOLUSDT", val_start, test_start)
    m = built["metrics"]
    assert built["train_end"] < test_start             # refit on train + validation only
    assert m["n_test"] == (data.loc[data["symbol"] == "SOLUSDT", "open_time"] >= test_start).sum()
    assert m["test_auc"] > 0.6


def test_shared_model_reports_each_symbol():
    data = make_dataset()
    built = tm.build(data, None, *tm.time_cutoffs(data))
    assert {"test_auc_BTCUSDT", "test_auc_SOLUSDT"} <= set(built["metrics"])


def test_version_is_safe_as_a_file_name():
    version = tm.make_version(datetime(2026, 10, 30, 9, 15, 2, tzinfo=timezone.utc))
    assert version.startswith("20261030T091502Z-")
    assert not set(version) & set(':\\/*?"<>|')


# --- Artifact and predict ---------------------------------------------------
@pytest.fixture
def artifact(tmp_path, monkeypatch):
    """A trained logreg saved the way train_model saves it, inside a temporary repo root."""
    monkeypatch.setattr(tm, "ROOT", tmp_path)
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    pm.load_artifact.cache_clear()
    data = make_dataset(symbols=("BTCUSDT",))
    model = tm.make_model("logreg").fit(data[tm.FEATURE_COLS], data[tm.TARGET])
    path = tm.save_artifact(model, "logreg", "BTCUSDT", "20261030T091502Z-test")
    yield path, model, data
    pm.load_artifact.cache_clear()


def test_artifact_roundtrip_and_score(artifact):
    path, model, data = artifact
    assert path == "models/direction/BTCUSDT/20261030T091502Z-test.joblib"
    loaded = pm.load_artifact(path)
    assert loaded["feature_cols"] == tm.FEATURE_COLS and loaded["feature_set"] == tm.FEATURE_SET
    row = data.iloc[-1]
    expected = model.predict_proba(data[tm.FEATURE_COLS].iloc[[-1]])[0, 1]
    # score() must not depend on the order of the dict it gets
    shuffled = {c: row[c] for c in reversed(tm.FEATURE_COLS)}
    assert pm.score(loaded, shuffled) == pytest.approx(expected)


def test_score_survives_a_missing_feature(artifact):
    path, _, data = artifact
    features = data.iloc[-1][tm.FEATURE_COLS].to_dict()
    features["body_rel"] = None                        # e.g. a candle with high == low
    assert 0 < pm.score(pm.load_artifact(path), features) < 1


def test_missing_artifact_file_gives_a_clear_error(artifact):
    with pytest.raises(pm.PredictionError, match="not in git"):
        pm.load_artifact("models/direction/BTCUSDT/does-not-exist.joblib")


class FakeConnection:
    def close(self):
        self.closed = True


def fake_db(monkeypatch, path, data, feature_set=tm.FEATURE_SET):
    """Replace the three database steps of predict(); returns the list of saved predictions."""
    saved = []
    open_time = datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(pm, "get_pg_connection", FakeConnection)
    monkeypatch.setattr(pm, "load_active_model", lambda conn, symbol, interval: {
        "symbol_id": 1, "model_id": 7, "model_version": "20261030T091502Z-test",
        "artifact_path": path, "feature_set": feature_set})
    monkeypatch.setattr(pm, "load_latest_features", lambda conn, sid, cols, interval: (
        open_time, feature_set, data.iloc[-1][cols].to_dict()))
    monkeypatch.setattr(pm, "save_prediction", lambda conn, entry, ot, prob, horizon, interval, source: saved.append(
        (entry["model_id"], ot, prob, horizon, source)))
    return saved


def test_predict_follows_the_api_contract(artifact, monkeypatch):
    path, _, data = artifact
    saved = fake_db(monkeypatch, path, data)
    result = pm.predict("btcusdt")
    assert set(result) == {"symbol", "open_time", "prob_up", "predicted_up", "model_version"}
    assert result["symbol"] == "BTCUSDT"
    assert result["open_time"] == "2026-10-30T08:00:00+00:00"
    assert 0 <= result["prob_up"] <= 1 and isinstance(result["predicted_up"], bool)
    assert result["predicted_up"] == (result["prob_up"] > 0.5)
    assert len(saved) == 1 and saved[0][0] == 7 and saved[0][3] == 1 and saved[0][4] == "adhoc"


def test_batch_run_predicts_every_symbol_and_skips_failures(artifact, monkeypatch):
    path, _, data = artifact
    saved = fake_db(monkeypatch, path, data)
    monkeypatch.setattr(pm, "symbols_with_model", lambda conn, interval: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    real_load = pm.load_active_model

    def load(conn, symbol, interval):
        if symbol == "ETHUSDT":
            raise pm.ModelNotFound("no model")
        return real_load(conn, symbol, interval)

    monkeypatch.setattr(pm, "load_active_model", load)
    monkeypatch.setattr(FakeConnection, "rollback", lambda self: None, raising=False)
    results, errors = pm.predict_all()
    assert [r["symbol"] for r in results] == ["BTCUSDT", "SOLUSDT"]
    assert set(errors) == {"ETHUSDT"}
    assert [s[4] for s in saved] == ["batch", "batch"]


def test_predict_without_saving(artifact, monkeypatch):
    path, _, data = artifact
    saved = fake_db(monkeypatch, path, data)
    pm.predict("BTCUSDT", save=False)
    assert saved == []


def test_predict_refuses_a_changed_feature_set(artifact, monkeypatch):
    path, _, data = artifact
    fake_db(monkeypatch, path, data, feature_set="v2")
    with pytest.raises(pm.PredictionError, match="Retrain"):
        pm.predict("BTCUSDT")


def test_response_time_is_utc():
    berlin = datetime(2026, 10, 30, 10, 0, tzinfo=timezone(pd.Timedelta(hours=1)))
    assert pm.to_response("BTCUSDT", berlin, 0.4, "v")["open_time"] == "2026-10-30T09:00:00+00:00"
