"""Unit tests for src/models/ (training and predictions, both tasks), no database needed."""
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
VOL_TARGET = tm.TASKS["volatility"]["target"]


def make_dataset(n=3000, symbols=("BTCUSDT", "SOLUSDT"), seed=0, signal=1.5):
    """Training-shaped rows: every feature column + both targets.

    realized_dir leans on ret_1, and realized_vol follows rv_16 (the last 4 hours),
    so a working model must beat a coin flip and the persistence baseline.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-05-01", periods=n, freq="15min", tz="UTC")
    frames = []
    for symbol in symbols:
        f = pd.DataFrame(rng.normal(size=(n, len(tm.FEATURE_COLS))), columns=tm.FEATURE_COLS)
        p_up = 1 / (1 + np.exp(-signal * f["ret_1"]))
        f[tm.TARGET] = (rng.uniform(size=n) < p_up).astype(int)
        f["rv_16"] = np.exp(rng.normal(-6, 0.5, n))                       # positive, like a real volatility
        f[VOL_TARGET] = 0.8 * f["rv_16"] * np.exp(rng.normal(0, 0.3, n))  # lower on average, plus noise
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


# --- Direction: models and metrics ------------------------------------------
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


# --- Volatility: models and metrics -----------------------------------------
def test_evaluate_volatility_perfect_and_off():
    y = np.array([0.001, 0.002, 0.004])
    perfect = tm.evaluate_volatility(y, y)
    assert perfect["rmse"] == 0 and perfect["mae"] == 0 and perfect["corr"] == pytest.approx(1)
    double = tm.evaluate_volatility(y, 2 * y)
    assert double["mae"] == pytest.approx(y.mean()) and double["corr"] == pytest.approx(1)


def test_volatility_baseline_is_the_last_four_hours():
    data = make_dataset(symbols=("BTCUSDT",))
    train, val, _ = tm.split_by_time(data, *tm.time_cutoffs(data), horizon=4)
    assert (tm.baseline_output(train, val, "volatility") == val["rv_16"].to_numpy()).all()


def test_volatility_models_beat_the_baseline():
    data = make_dataset(symbols=("BTCUSDT",))
    train, val, _ = tm.split_by_time(data, *tm.time_cutoffs(data), horizon=4)
    comparison, fitted = tm.compare_models(train, val, task="volatility")
    best = tm.pick_best(comparison, "volatility")
    assert best in ("ridge", "hgb_reg")                    # picked by the LOWEST rmse
    assert comparison.loc[best, "val_rmse"] < comparison.loc["baseline", "val_rmse"]
    # predictions are volatilities again (exp of the log the model learned): positive and realistic
    pred = tm.model_output(fitted[best], val[tm.FEATURE_COLS], "volatility")
    assert (pred > 0).all() and pred.mean() == pytest.approx(val[VOL_TARGET].mean(), rel=0.2)


def test_volatility_build_reports_improvement_over_baseline():
    data = make_dataset()
    built = tm.build(data, "BTCUSDT", *tm.time_cutoffs(data), task="volatility")
    m = built["metrics"]
    assert built["task"] == "volatility"
    assert m["improvement"] == pytest.approx(1 - m["test_rmse"] / m["baseline_rmse"])
    assert m["improvement"] > 0
    assert tm.summary(built)["task"] == "volatility"


# --- Artifact and predictions -----------------------------------------------
@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """One trained model per task, saved the way train_model saves them, inside a temporary repo root."""
    monkeypatch.setattr(tm, "ROOT", tmp_path)
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    pm.load_artifact.cache_clear()
    data = make_dataset(symbols=("BTCUSDT",))
    paths, models = {}, {}
    for task, name in (("direction", "logreg"), ("volatility", "ridge")):
        models[task] = tm.fit_model(name, data, task)
        paths[task] = tm.save_artifact(models[task], name, "BTCUSDT", "20261030T091502Z-test", task=task)
    yield paths, models, data
    pm.load_artifact.cache_clear()


def test_artifact_roundtrip_and_score(artifacts):
    paths, models, data = artifacts
    assert paths["direction"] == "models/direction/BTCUSDT/20261030T091502Z-test.joblib"
    assert paths["volatility"] == "models/volatility/BTCUSDT/20261030T091502Z-test.joblib"
    loaded = pm.load_artifact(paths["direction"])
    assert loaded["feature_cols"] == tm.FEATURE_COLS and loaded["feature_set"] == tm.FEATURE_SET
    row = data.iloc[-1]
    expected = models["direction"].predict_proba(data[tm.FEATURE_COLS].iloc[[-1]])[0, 1]
    # score() must not depend on the order of the dict it gets
    shuffled = {c: row[c] for c in reversed(tm.FEATURE_COLS)}
    assert pm.score(loaded, shuffled) == pytest.approx(expected)


def test_volatility_score_is_turned_back_from_the_logarithm(artifacts):
    paths, models, data = artifacts
    loaded = pm.load_artifact(paths["volatility"])
    assert loaded["log_target"] is True and loaded["horizon_candles"] == 4
    log_prediction = models["volatility"].predict(data[tm.FEATURE_COLS].iloc[[-1]])[0]
    value = pm.score(loaded, data.iloc[-1][tm.FEATURE_COLS].to_dict())
    assert value == pytest.approx(np.exp(log_prediction)) and value > 0


def test_score_survives_a_missing_feature(artifacts):
    paths, _, data = artifacts
    features = data.iloc[-1][tm.FEATURE_COLS].to_dict()
    features["body_rel"] = None                        # e.g. a candle with high == low
    assert 0 < pm.score(pm.load_artifact(paths["direction"]), features) < 1


def test_missing_artifact_file_gives_a_clear_error(artifacts):
    with pytest.raises(pm.PredictionError, match="not in git"):
        pm.load_artifact("models/direction/BTCUSDT/does-not-exist.joblib")


def test_model_from_another_scikit_learn_version_is_refused(artifacts, tmp_path):
    paths, _, _ = artifacts
    import joblib
    saved = joblib.load(tmp_path / paths["direction"])
    saved["lib_versions"]["scikit-learn"] = "0.0.1"
    joblib.dump(saved, tmp_path / paths["direction"])
    pm.load_artifact.cache_clear()
    with pytest.raises(pm.PredictionError, match="saved with scikit-learn 0.0.1"):
        pm.load_artifact(paths["direction"])


class FakeConnection:
    def close(self):
        pass

    def rollback(self):
        pass


def fake_db(monkeypatch, paths, data, feature_set=tm.FEATURE_SET, symbols=("BTCUSDT",)):
    """Replace the database steps of predict(); returns the list of saved predictions."""
    saved = []
    open_time = datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)

    def load_active_model(conn, symbol, task, interval):
        if symbol not in symbols:
            raise pm.ModelNotFound("no model")
        return {"symbol_id": 1, "model_id": 7, "model_version": "20261030T091502Z-test",
                "artifact_path": paths[task], "feature_set": feature_set}

    def save_prediction(conn, entry, ot, value, horizon, interval, task, source, baseline):
        saved.append({"model_id": entry["model_id"], "open_time": ot, "value": value, "horizon": horizon,
                      "task": task, "source": source, "baseline": baseline})

    monkeypatch.setattr(pm, "get_pg_connection", FakeConnection)
    monkeypatch.setattr(pm, "load_active_model", load_active_model)
    monkeypatch.setattr(pm, "load_latest_features", lambda conn, sid, cols, interval: (
        open_time, feature_set, data.iloc[-1][cols].to_dict()))
    monkeypatch.setattr(pm, "save_prediction", save_prediction)
    return saved


def test_predict_direction_response_and_saved_row(artifacts, monkeypatch):
    paths, _, data = artifacts
    saved = fake_db(monkeypatch, paths, data)
    result = pm.predict("btcusdt")
    assert set(result) == {"symbol", "task", "open_time", "prob_up", "predicted_up", "model_version"}
    assert result["symbol"] == "BTCUSDT" and result["task"] == "direction"
    assert result["open_time"] == "2026-10-30T08:00:00+00:00"
    assert 0 <= result["prob_up"] <= 1 and isinstance(result["predicted_up"], bool)
    assert result["predicted_up"] == (result["prob_up"] > 0.5)
    assert len(saved) == 1
    assert saved[0]["model_id"] == 7 and saved[0]["horizon"] == 1 and saved[0]["source"] == "adhoc"
    assert saved[0]["baseline"] == 0.5


def test_predict_volatility_response_and_saved_row(artifacts, monkeypatch):
    paths, _, data = artifacts
    saved = fake_db(monkeypatch, paths, data)
    result = pm.predict("BTCUSDT", task="volatility")
    assert set(result) == {"symbol", "task", "open_time", "predicted_vol", "model_version"}
    assert result["predicted_vol"] > 0
    # the API and v_prediction_quality expect: horizon 4, the last-4-hours volatility as baseline
    assert saved[0]["task"] == "volatility" and saved[0]["horizon"] == 4
    assert saved[0]["baseline"] == pytest.approx(data.iloc[-1]["rv_16"])


def test_batch_run_covers_every_task_and_skips_failures(artifacts, monkeypatch):
    paths, _, data = artifacts
    saved = fake_db(monkeypatch, paths, data, symbols=("BTCUSDT", "SOLUSDT"))
    monkeypatch.setattr(pm, "symbols_with_model", lambda conn, task, interval: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    results, errors = pm.predict_all()
    assert [(r["task"], r["symbol"]) for r in results] == [
        ("direction", "BTCUSDT"), ("direction", "SOLUSDT"), ("volatility", "BTCUSDT"), ("volatility", "SOLUSDT")]
    assert set(errors) == {"direction/ETHUSDT", "volatility/ETHUSDT"}
    assert {s["source"] for s in saved} == {"batch"}


def test_predict_without_saving(artifacts, monkeypatch):
    paths, _, data = artifacts
    saved = fake_db(monkeypatch, paths, data)
    pm.predict("BTCUSDT", save=False)
    assert saved == []


def test_predict_refuses_a_changed_feature_set(artifacts, monkeypatch):
    paths, _, data = artifacts
    fake_db(monkeypatch, paths, data, feature_set="v2")
    with pytest.raises(pm.PredictionError, match="Retrain"):
        pm.predict("BTCUSDT")


def test_only_direction_gets_an_up_down_label():
    executed = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            executed.append(params)

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

    entry = {"symbol_id": 1, "model_id": 7}
    pm.save_prediction(Conn(), entry, "t", 0.7, 1, task="direction")
    pm.save_prediction(Conn(), entry, "t", 0.002, 4, task="volatility", baseline=0.003)
    assert executed[0]["label"] == 1 and executed[1]["label"] is None
    assert executed[1]["value"] == 0.002 and executed[1]["baseline"] == 0.003


def test_response_time_is_utc():
    berlin = datetime(2026, 10, 30, 10, 0, tzinfo=timezone(pd.Timedelta(hours=1)))
    assert pm.to_response("BTCUSDT", berlin, 0.4, "v")["open_time"] == "2026-10-30T09:00:00+00:00"
