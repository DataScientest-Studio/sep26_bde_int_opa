"""MODEL: train the 'direction' model: will the next 15m candle close higher?

Owner: Dennis  (Stage 2)

What it does:
1. load_dataset(): features + outcomes from gold via sql/queries/training_dataset.sql
2. split_by_time(): chronological train / validation / test split, the same dates for
   every symbol, with a gap ("purge") so no training label looks into the next part
3. compare_models(): a baseline and two models, judged on the validation part only
4. finalize(): refit the chosen model on train + validation, score it once on test
5. save_artifact() + register_model(): write models/<task>/<symbol>/<version>.joblib
   and one row in model_registry; activate() makes it the model predict() uses

The notebook notebooks/01_train_direction.ipynb walks through these steps with
explanations. This file can also do everything in one go, e.g. for scheduled
retraining later:

Run from the repo root:
    python src/models/train_model.py                  # BTCUSDT + SOLUSDT, 15m, next candle
    python src/models/train_model.py --activate       # also make the new models active
    python src/models/train_model.py --shared         # plus one shared model for both symbols
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from psycopg2.extras import Json
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))
from build_features import FEATURE_COLS, FEATURE_SET  # noqa: E402  (the gold contract, never a copy)
from db import get_pg_connection  # noqa: E402

TASK = "direction"
INTERVAL = "15m"
HORIZON = 1                     # candles ahead: 1 = the next 15 minutes
TARGET = "realized_dir"         # 1 if the price went up over the horizon, else 0
SYMBOLS = ("BTCUSDT", "SOLUSDT")
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # the rest (15 %) is the test part
CANDLE = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
          "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}

# The candidates. Deliberately small and strongly regularised: 15-minute returns are
# mostly noise, and a flexible model would just memorise it.
MODELS = {
    "baseline": {
        "algo": "sklearn.DummyClassifier",
        "params": {"strategy": "prior"},          # always answers "share of up candles in training"
    },
    "logreg": {
        "algo": "sklearn.LogisticRegression",
        "params": {"C": 0.1, "max_iter": 1000},
    },
    "hgb": {
        "algo": "sklearn.HistGradientBoostingClassifier",
        "params": {"learning_rate": 0.05, "max_iter": 200, "max_depth": 3, "min_samples_leaf": 200,
                   "l2_regularization": 1.0, "early_stopping": False, "random_state": 0},
    },
}

TRAINING_SQL = (ROOT / "sql" / "queries" / "training_dataset.sql").read_text()

REGISTER_SQL = """
INSERT INTO model_registry (task, symbol_id, interval, model_version, algo, feature_set, feature_cols,
                            horizon_candles, params, metrics, train_start, train_end, artifact_path,
                            lib_versions)
VALUES (%(task)s, (SELECT symbol_id FROM dim_symbol WHERE symbol = %(symbol)s), %(interval)s,
        %(model_version)s, %(algo)s, %(feature_set)s, %(feature_cols)s, %(horizon)s, %(params)s,
        %(metrics)s, %(train_start)s, %(train_end)s, %(artifact_path)s, %(lib_versions)s)
RETURNING model_id;
"""


# --- Data -------------------------------------------------------------------
def load_dataset(conn, symbol, interval=INTERVAL, horizon=HORIZON):
    """Features + outcome per candle for one symbol, oldest first.

    The first ~96 candles have no complete history yet (e.g. the 1-day return needs
    96 earlier candles), so their features are empty. Those rows are dropped.
    """
    with conn.cursor() as cur:
        cur.execute(TRAINING_SQL, {"symbol": symbol, "interval": interval, "horizon": horizon})
        df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    return df.dropna(subset=FEATURE_COLS).sort_values("open_time").reset_index(drop=True)


def load_symbols(conn, symbols=SYMBOLS, interval=INTERVAL, horizon=HORIZON):
    """load_dataset() for several symbols, stacked into one frame."""
    frames = [load_dataset(conn, s, interval, horizon) for s in symbols]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No training data. Run src/features/build_features.py first.")
    return pd.concat(frames, ignore_index=True)


# --- Chronological split ----------------------------------------------------
def time_cutoffs(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Start of the validation part and of the test part, as timestamps.

    Computed from all symbols together, so BTC and SOL are cut on the same dates.
    Otherwise a shared model (and SOL's BTC features) could see the test period
    through the other coin.
    """
    times = pd.DatetimeIndex(df["open_time"]).unique().sort_values()   # keeps the UTC time zone
    return times[int(len(times) * train_frac)], times[int(len(times) * (train_frac + val_frac))]


def split_by_time(df, val_start, test_start, horizon=HORIZON, interval=INTERVAL):
    """Train / validation / test, never shuffled.

    The label of candle t is decided by the close of candle t + horizon. A training row
    whose label candle lies in the validation period would leak validation prices
    into training, so those last `horizon` rows before each cut are left out ("purge").
    """
    label_time = df["open_time"] + horizon * CANDLE[interval]
    train = df[label_time < val_start]
    val = df[(df["open_time"] >= val_start) & (label_time < test_start)]
    test = df[df["open_time"] >= test_start]
    return train, val, test


# --- Models and metrics -----------------------------------------------------
def make_model(name):
    """A fresh, untrained estimator for one entry of MODELS."""
    params = MODELS[name]["params"]
    if name == "baseline":
        return DummyClassifier(**params)
    if name == "logreg":
        # median-fill missing values (rare, e.g. a candle with high == low), then put all
        # features on the same scale: RSI runs from 0 to 100, returns are around 0.001
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(**params))
    if name == "hgb":
        return HistGradientBoostingClassifier(**params)   # handles missing values by itself
    raise ValueError(f"unknown model {name}")


def evaluate(y_true, prob_up, threshold=0.5):
    """The numbers we report for a set of predictions.

    auc       how well "up" candles are ranked above "down" candles: 0.5 = coin flip, 1 = perfect
    accuracy  share of correct up/down calls at the 0.5 threshold
    log_loss  how good the probabilities are (lower is better; ~0.693 = always saying 50 %)
    """
    y_true = np.asarray(y_true).astype(int)
    prob_up = np.clip(np.asarray(prob_up, dtype=float), 1e-6, 1 - 1e-6)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "up_share": float(y_true.mean()) if len(y_true) else float("nan"),
        "auc": float(roc_auc_score(y_true, prob_up)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, prob_up > threshold)),
        "log_loss": float(log_loss(y_true, prob_up, labels=[0, 1])),
    }


def compare_models(train, val, names=("baseline", "logreg", "hgb")):
    """Fit every candidate on train, score it on validation. Test stays untouched."""
    rows, fitted = [], {}
    for name in names:
        model = make_model(name).fit(train[FEATURE_COLS], train[TARGET])
        scores = evaluate(val[TARGET], model.predict_proba(val[FEATURE_COLS])[:, 1])
        rows.append({"model": name, **{f"val_{k}": v for k, v in scores.items()}})
        fitted[name] = model
    return pd.DataFrame(rows).set_index("model"), fitted


def pick_best(comparison):
    """The real model (not the baseline) with the highest validation AUC."""
    candidates = comparison.drop(index="baseline", errors="ignore")
    return candidates["val_auc"].idxmax()


def finalize(name, train, val, test, symbol_filter=None):
    """Refit the chosen model on train + validation and score it once on test.

    symbol_filter: for a shared model, also report the test scores per symbol.
    """
    fit_on = pd.concat([train, val])
    model = make_model(name).fit(fit_on[FEATURE_COLS], fit_on[TARGET])
    prob = model.predict_proba(test[FEATURE_COLS])[:, 1]
    metrics = {f"test_{k}": v for k, v in evaluate(test[TARGET], prob).items()}
    # baseline on the same test rows: always say "up" with the training up-share
    majority = int(fit_on[TARGET].mean() >= 0.5)
    metrics["baseline_auc"] = 0.5
    metrics["baseline_accuracy"] = float((test[TARGET] == majority).mean())
    if symbol_filter:
        for symbol in symbol_filter:
            part = test["symbol"] == symbol
            metrics[f"test_auc_{symbol}"] = evaluate(test.loc[part, TARGET], prob[part.to_numpy()])["auc"]
    metrics["n_train"] = int(len(fit_on))
    return model, metrics, fit_on["open_time"].min(), fit_on["open_time"].max()


# --- Artifacts and registry -------------------------------------------------
def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def make_version(now=None):
    """e.g. 20261030T091502Z-6821507: training time (UTC) + the code's git commit.

    No colons, because the version is part of a file name and Windows forbids ':'.
    """
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{git_sha()}"


def lib_versions():
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}


def save_artifact(model, name, symbol, version, interval=INTERVAL, horizon=HORIZON):
    """Write the model plus everything predict() needs to use it correctly.

    symbol=None means a shared model for all symbols (folder ALL).
    Returns the path relative to the repo root (that is what model_registry stores).
    """
    rel_path = Path("models") / TASK / (symbol or "ALL") / f"{version}.joblib"
    (ROOT / rel_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "name": name,
        "algo": MODELS[name]["algo"],
        "task": TASK,
        "symbol": symbol,
        "interval": interval,
        "horizon_candles": horizon,
        "feature_set": FEATURE_SET,
        "feature_cols": list(FEATURE_COLS),     # the exact order the model was trained with
        "model_version": version,
        "lib_versions": lib_versions(),
    }, ROOT / rel_path)
    return rel_path.as_posix()


def register_model(conn, name, symbol, version, artifact_path, metrics, train_start, train_end,
                   interval=INTERVAL, horizon=HORIZON):
    """One row in model_registry. Not active yet: see activate(). Returns model_id."""
    with conn.cursor() as cur:
        cur.execute(REGISTER_SQL, {
            "task": TASK, "symbol": symbol, "interval": interval, "model_version": version,
            "algo": MODELS[name]["algo"], "feature_set": FEATURE_SET, "feature_cols": Json(list(FEATURE_COLS)),
            "horizon": horizon, "params": Json(MODELS[name]["params"]), "metrics": Json(_clean(metrics)),
            "train_start": train_start, "train_end": train_end, "artifact_path": artifact_path,
            "lib_versions": Json(lib_versions()),
        })
        model_id = cur.fetchone()[0]
    conn.commit()
    return model_id


def activate(conn, model_id):
    """Make this model the one predict() uses for its task, symbol and interval.

    The previously active one is switched off in the same transaction; the unique
    index model_registry_one_active guarantees there is never more than one.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE model_registry old SET is_active = FALSE
            FROM model_registry new
            WHERE new.model_id = %(id)s AND old.is_active AND old.model_id <> new.model_id
              AND old.task = new.task AND old.interval = new.interval
              AND COALESCE(old.symbol_id, 0) = COALESCE(new.symbol_id, 0);
            UPDATE model_registry SET is_active = TRUE, promoted_at = NOW() WHERE model_id = %(id)s;
        """, {"id": model_id})
    conn.commit()


def _clean(metrics):
    """NaN is not valid JSON: store it as null."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}


# --- Everything in one go ---------------------------------------------------
def build(data, symbol, val_start, test_start):
    """Split, compare on validation, pick the best, refit and test it. Nothing is saved yet.

    symbol=None builds a shared model on all symbols in `data`.
    """
    part = data if symbol is None else data[data["symbol"] == symbol]
    train, val, test = split_by_time(part, val_start, test_start)
    comparison, _ = compare_models(train, val)
    best = pick_best(comparison)
    model, metrics, train_start, train_end = finalize(
        best, train, val, test, symbol_filter=sorted(part["symbol"].unique()) if symbol is None else None)
    metrics.update({f"val_auc_{m}": float(comparison.loc[m, "val_auc"]) for m in comparison.index})
    metrics.update({"n_val": int(len(val)), "n_test": int(len(test)),
                    "val_start": str(val_start), "test_start": str(test_start)})
    return {"symbol": symbol, "name": best, "model": model, "metrics": metrics, "comparison": comparison,
            "train_start": train_start, "train_end": train_end}


def persist(conn, built, activate_model=False):
    """Save the artifact of a build() result, register it, optionally activate it."""
    version = make_version()
    path = save_artifact(built["model"], built["name"], built["symbol"], version)
    model_id = register_model(conn, built["name"], built["symbol"], version, path, built["metrics"],
                              built["train_start"], built["train_end"])
    if activate_model:
        activate(conn, model_id)
    return {"model_id": model_id, "model_version": version, "artifact_path": path, "active": activate_model}


def summary(built, saved=None):
    """One line per model for printing."""
    m = built["metrics"]
    line = {"symbol": built["symbol"] or "ALL", "model": built["name"],
            "val_auc": float(built["comparison"].loc[built["name"], "val_auc"]),
            "test_auc": m["test_auc"], "test_accuracy": m["test_accuracy"],
            "baseline_accuracy": m["baseline_accuracy"], "n_test": m["n_test"]}
    return {**line, **(saved or {})}


def train_and_register(conn, data, symbol, val_start, test_start, activate_model=False):
    """build() + persist() for one symbol (None = shared model)."""
    built = build(data, symbol, val_start, test_start)
    return summary(built, persist(conn, built, activate_model))


def main():
    parser = argparse.ArgumentParser(description="Train and register direction models.")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    parser.add_argument("--shared", action="store_true", help="also train one model for all symbols")
    parser.add_argument("--activate", action="store_true", help="make the new models active")
    args = parser.parse_args()

    conn = get_pg_connection()
    try:
        data = load_symbols(conn, args.symbols)
        val_start, test_start = time_cutoffs(data)
        print(f"{len(data)} rows; validation from {val_start:%Y-%m-%d %H:%M}, test from {test_start:%Y-%m-%d %H:%M} UTC")
        targets = list(args.symbols) + ([None] if args.shared else [])
        results = [train_and_register(conn, data, s, val_start, test_start, args.activate) for s in targets]
    finally:
        conn.close()
    print(pd.DataFrame(results).set_index("symbol").to_string(float_format=lambda x: f"{x:.4f}"))
    print(json.dumps({"registered": [r["model_id"] for r in results]}))


if __name__ == "__main__":
    main()
