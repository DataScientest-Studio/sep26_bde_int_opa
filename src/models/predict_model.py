"""MODEL: predictions of the 'direction' model: will the next candle close higher?

Owner: Dennis  (Stage 2)

How it fits the API: the API never loads models. This script writes predictions into
the `predictions` table, and GET /predict/{symbol} reads the newest one of the active
model from there (view v_latest_predictions). Run it after every build_features.py run
(Airflow, Stage 3), otherwise the API marks the answer as `stale`.

What predict(symbol) does:
1. finds the active 'direction' model in model_registry: the symbol's own model,
   otherwise a shared one (symbol_id NULL)
2. loads its artifact (models/direction/<symbol>/<version>.joblib, cached in memory)
3. reads the newest row of gold_features for the symbol
4. asks the model for P(up), using exactly the feature columns it was trained with
5. stores the prediction in `predictions`, unless save=False, and returns
       {"symbol": "BTCUSDT", "open_time": "2026-10-19T09:00:00+00:00",
        "prob_up": 0.57, "predicted_up": True, "model_version": "..."}
   open_time is the candle the features come from; the prediction is about the candle after it.

Errors:
    ModelNotFound     no active model for this symbol (train one first)
    FeaturesNotFound  no gold_features row for this symbol yet
    PredictionError   anything else that makes a prediction impossible

Run from the repo root:
    python src/models/predict_model.py --all          # every symbol with an active model (source 'batch')
    python src/models/predict_model.py BTCUSDT        # one symbol (source 'adhoc')
    python src/models/predict_model.py SOLUSDT --no-save
"""
import argparse
import json
import sys
from contextlib import closing
from datetime import timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
from db import get_pg_connection  # noqa: E402

TASK = "direction"
INTERVAL = "15m"
THRESHOLD = 0.5        # prob_up above this counts as "up"
BASELINE = 0.5         # the naive forecast stored next to every prediction


class PredictionError(Exception):
    """A prediction is not possible right now."""


class ModelNotFound(PredictionError):
    """No active model for this symbol."""


class FeaturesNotFound(PredictionError):
    """No features for this symbol in gold_features."""


ACTIVE_MODEL_SQL = """
SELECT s.symbol_id, r.model_id, r.model_version, r.artifact_path, r.feature_set
FROM dim_symbol s
JOIN model_registry r
  ON r.task = %(task)s AND r.interval = %(interval)s AND r.is_active
 AND (r.symbol_id = s.symbol_id OR r.symbol_id IS NULL)
WHERE s.symbol = %(symbol)s
ORDER BY r.symbol_id IS NULL, r.trained_at DESC     -- the symbol's own model before a shared one
LIMIT 1;
"""

SAVE_SQL = """
INSERT INTO predictions (symbol_id, interval, open_time, task, model_id, horizon_candles,
                         y_pred_label, y_pred_value, y_baseline, source)
VALUES (%(symbol_id)s, %(interval)s, %(open_time)s, %(task)s, %(model_id)s, %(horizon)s,
        %(label)s, %(prob)s, %(baseline)s, %(source)s)
ON CONFLICT (symbol_id, interval, open_time, task, model_id) DO UPDATE SET
    y_pred_label = EXCLUDED.y_pred_label, y_pred_value = EXCLUDED.y_pred_value,
    y_baseline = EXCLUDED.y_baseline, source = EXCLUDED.source, predicted_at = NOW();
"""

# symbols that have an active model of their own, or all symbols if a shared model is active
SYMBOLS_WITH_MODEL_SQL = """
SELECT s.symbol FROM dim_symbol s
WHERE EXISTS (SELECT 1 FROM model_registry r
              WHERE r.task = %(task)s AND r.interval = %(interval)s AND r.is_active
                AND (r.symbol_id = s.symbol_id OR r.symbol_id IS NULL))
ORDER BY s.symbol;
"""


# --- Database ---------------------------------------------------------------
def load_active_model(conn, symbol, task=TASK, interval=INTERVAL):
    """The registry row of the model to use for this symbol."""
    with conn.cursor() as cur:
        cur.execute(ACTIVE_MODEL_SQL, {"symbol": symbol, "task": task, "interval": interval})
        row = cur.fetchone()
    if row is None:
        raise ModelNotFound(f"No active {task} model for {symbol} ({interval}). "
                            "Train one: python src/models/train_model.py --activate")
    keys = ("symbol_id", "model_id", "model_version", "artifact_path", "feature_set")
    return dict(zip(keys, row))


def load_latest_features(conn, symbol_id, feature_cols, interval=INTERVAL):
    """Newest gold_features row: (open_time, feature_set, {column: value})."""
    query = sql.SQL("""
        SELECT open_time, feature_set, {cols} FROM gold_features
        WHERE symbol_id = %(symbol_id)s AND interval = %(interval)s
        ORDER BY open_time DESC LIMIT 1;
    """).format(cols=sql.SQL(", ").join(sql.Identifier(c) for c in feature_cols))
    with conn.cursor() as cur:
        cur.execute(query, {"symbol_id": symbol_id, "interval": interval})
        row = cur.fetchone()
    if row is None:
        raise FeaturesNotFound(f"No gold_features for symbol_id {symbol_id} ({interval}). "
                               "Run: python src/features/build_features.py")
    return row[0], row[1], dict(zip(feature_cols, row[2:]))


def save_prediction(conn, entry, open_time, prob, horizon, interval=INTERVAL, task=TASK, source="adhoc"):
    with conn.cursor() as cur:
        cur.execute(SAVE_SQL, {
            "symbol_id": entry["symbol_id"], "interval": interval, "open_time": open_time, "task": task,
            "model_id": entry["model_id"], "horizon": horizon, "label": int(prob > THRESHOLD),
            "prob": prob, "baseline": BASELINE, "source": source,
        })
    conn.commit()


def symbols_with_model(conn, task=TASK, interval=INTERVAL):
    with conn.cursor() as cur:
        cur.execute(SYMBOLS_WITH_MODEL_SQL, {"task": task, "interval": interval})
        return [r[0] for r in cur.fetchall()]


# --- The model --------------------------------------------------------------
@lru_cache(maxsize=16)
def load_artifact(artifact_path):
    """The saved model and its metadata. Cached: the API loads each file only once."""
    path = ROOT / artifact_path
    if not path.is_file():
        raise PredictionError(f"Model file {artifact_path} is missing on this machine. "
                              "Artifacts are not in git: train locally with src/models/train_model.py")
    return joblib.load(path)


def score(artifact, features):
    """P(up) for one row of features, in the column order the model was trained with."""
    cols = artifact["feature_cols"]
    row = pd.DataFrame([[features.get(c) for c in cols]], columns=cols).astype(float)   # None -> NaN
    prob = float(artifact["model"].predict_proba(row)[0, 1])
    if not np.isfinite(prob):
        raise PredictionError("The model returned no valid probability.")
    return prob


def to_response(symbol, open_time, prob, model_version):
    """The dict promised to the API (see the contract at the top)."""
    ts = pd.Timestamp(open_time)
    ts = ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
    return {
        "symbol": symbol,
        "open_time": ts.isoformat(),
        "prob_up": round(prob, 4),
        "predicted_up": prob > THRESHOLD,
        "model_version": model_version,
    }


# --- Predictions --------------------------------------------------------------
def predict_with(conn, symbol, interval=INTERVAL, save=True, source="adhoc"):
    """predict() on an open connection (used by predict() and predict_all())."""
    entry = load_active_model(conn, symbol, interval=interval)
    artifact = load_artifact(entry["artifact_path"])
    open_time, feature_set, features = load_latest_features(
        conn, entry["symbol_id"], artifact["feature_cols"], interval)
    if feature_set != artifact["feature_set"]:
        # a formula in gold changed since training: the model would get numbers it never saw
        raise PredictionError(f"Model {entry['model_version']} was trained on feature set "
                              f"{artifact['feature_set']}, gold now has {feature_set}. Retrain the model.")
    prob = score(artifact, features)
    if save:
        save_prediction(conn, entry, open_time, prob, artifact["horizon_candles"], interval, source=source)
    return to_response(symbol, open_time, prob, entry["model_version"])


def predict(symbol, interval=INTERVAL, save=True):
    """Probability that the candle after the newest one closes higher. See the top of the file."""
    with closing(get_pg_connection()) as conn:
        return predict_with(conn, symbol.upper(), interval, save)


def predict_all(interval=INTERVAL, save=True):
    """The batch run: one prediction per symbol that has an active model (source 'batch').

    A symbol that fails (e.g. no features yet) is reported and skipped; the others still run.
    """
    results, errors = [], {}
    with closing(get_pg_connection()) as conn:
        for symbol in symbols_with_model(conn, interval=interval):
            try:
                results.append(predict_with(conn, symbol, interval, save, source="batch"))
            except PredictionError as error:
                conn.rollback()
                errors[symbol] = str(error)
    return results, errors


def main():
    parser = argparse.ArgumentParser(description="Predict the direction of the next candle.")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT")
    parser.add_argument("--all", action="store_true", help="every symbol with an active model (batch)")
    parser.add_argument("--no-save", action="store_true", help="do not write to the predictions table")
    args = parser.parse_args()
    try:
        if args.all:
            results, errors = predict_all(save=not args.no_save)
            print(json.dumps({"predictions": results, "errors": errors}, indent=2))
            if not results:
                raise SystemExit("No predictions written. " + " ".join(errors.values()))
        else:
            print(json.dumps(predict(args.symbol, save=not args.no_save), indent=2))
    except PredictionError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
