"""FastAPI application: read-only access to the warehouse (silver + gold).

Owner: Talgat  (Stage 2)

Design: the API never trains or loads models. The ML pipeline writes predictions
into the `predictions` table; the API reads the newest one of the active model.
Until a model is active, /predict answers with a transparent naive baseline
(`source: "baseline"`), so the contract stays the same when the model arrives.

All SQL lives in sql/queries/*.sql (one query per file).

Endpoints:
    GET /health                       liveness: the process is up
    GET /ready                        readiness: PostgreSQL answers
    GET /symbols                      trading pairs
    GET /candles?symbol=&interval=&limit=
    GET /stats                        summary per symbol and interval
    GET /features/latest?symbol=      newest gold feature row
    GET /predict/{symbol}?task=direction|volatility
    GET /predict?symbol=&task=        same, symbol as query parameter

Run from the repo root:
    uvicorn src.api.main:app --reload
    open http://localhost:8000/docs   (interactive Swagger UI)
"""
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "sql" / "queries"
sys.path.insert(0, str(ROOT / "src" / "data"))

SYMBOL_PATTERN = r"^[A-Z0-9]{5,20}$"
MAX_CANDLES = 2000


# ---------------------------------------------------------------- database access

def load_query(name: str) -> str:
    return (QUERIES / f"{name}.sql").read_text()


class Database:
    """Thin wrapper: run a named query from sql/queries and return rows as dicts."""

    def fetch_all(self, query_name: str, params: Optional[dict] = None) -> list:
        from psycopg2.extras import RealDictCursor  # imported lazily so tests need no driver
        from db import get_pg_connection

        conn = get_pg_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(load_query(query_name), params or {})
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def ping(self) -> bool:
        from db import get_pg_connection

        conn = get_pg_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return cur.fetchone()[0] == 1
        finally:
            conn.close()


def get_db() -> Database:
    return Database()


# ---------------------------------------------------------------- response models

class Task(str, Enum):
    direction = "direction"
    volatility = "volatility"


class Candle(BaseModel):
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int


class Stat(BaseModel):
    symbol: str
    interval: str
    candles: int
    first_candle: datetime
    last_candle: datetime
    min_price: float
    max_price: float
    avg_close: float
    total_volume: float


class Prediction(BaseModel):
    symbol: str
    task: Task
    open_time: datetime            # candle the prediction is based on
    horizon_candles: int           # 1 = next candle, 4 = next hour
    prob_up: Optional[float] = None          # direction only
    predicted_up: Optional[bool] = None      # direction only
    predicted_vol: Optional[float] = None    # volatility only (std of 15m log returns)
    model_version: str
    source: str                    # "model" (active model in DB) or "baseline"
    stale: bool = False            # True if the model's prediction is older than the newest candle


# ---------------------------------------------------------------- app

app = FastAPI(
    title="OPA API",
    description="Binance BTC/SOL candles, gold features and model predictions.",
    version="0.2.0",
)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready(db: Database = Depends(get_db)):
    try:
        db.ping()
    except Exception as exc:  # any driver/connection error means "not ready"
        raise HTTPException(status_code=503, detail=f"database not reachable: {type(exc).__name__}")
    return {"status": "ready"}


@app.get("/symbols", tags=["data"])
def symbols(db: Database = Depends(get_db)):
    return db.fetch_all("symbols")


@app.get("/candles", response_model=list[Candle], tags=["data"])
def candles(
    symbol: str = Query("BTCUSDT", pattern=SYMBOL_PATTERN),
    interval: str = Query("15m", pattern=r"^\d{1,2}[mhd]$"),
    limit: int = Query(96, ge=1, le=MAX_CANDLES),
    db: Database = Depends(get_db),
):
    """Newest `limit` closed candles, oldest first (ready for a chart)."""
    rows = db.fetch_all("candles", {"symbol": symbol, "interval": interval, "limit": limit})
    if not rows:
        raise HTTPException(status_code=404, detail=f"no candles for {symbol} {interval}")
    return list(reversed(rows))


@app.get("/stats", response_model=list[Stat], tags=["data"])
def stats(db: Database = Depends(get_db)):
    return db.fetch_all("stats")


@app.get("/features/latest", tags=["data"])
def latest_features(
    symbol: str = Query("BTCUSDT", pattern=SYMBOL_PATTERN),
    interval: str = Query("15m", pattern=r"^\d{1,2}[mhd]$"),
    db: Database = Depends(get_db),
):
    rows = db.fetch_all("latest_features", {"symbol": symbol, "interval": interval})
    if not rows:
        raise HTTPException(status_code=404, detail=f"no features for {symbol} {interval}")
    return rows[0]


def _predict(symbol: str, task: "Task", interval: str, db: "Database") -> dict:
    params = {"symbol": symbol, "interval": interval}
    feats = db.fetch_all("latest_features", params)
    if not feats:
        raise HTTPException(status_code=404, detail=f"no features for {symbol} {interval}")
    latest = feats[0]

    rows = db.fetch_all("latest_prediction", {**params, "task": task.value})
    if rows:
        r = rows[0]
        out = dict(symbol=symbol, task=task, open_time=r["open_time"], horizon_candles=r["horizon_candles"],
                   model_version=r["model_version"], source="model",
                   stale=r["open_time"] < latest["open_time"])   # batch job is behind the newest candle
        if task is Task.direction:
            label = r["y_pred_label"]
            out.update(prob_up=r["y_pred_value"], predicted_up=None if label is None else bool(label))
        else:
            out.update(predicted_vol=r["y_pred_value"])
        return out

    if task is Task.direction:
        base = db.fetch_all("baseline_direction", params)
        prob_up = base[0]["prob_up"] if base else None
        return dict(symbol=symbol, task=task, open_time=latest["open_time"], horizon_candles=1,
                    prob_up=None if prob_up is None else round(prob_up, 4),
                    predicted_up=None if prob_up is None else prob_up >= 0.5,
                    model_version="baseline-upshare-96", source="baseline", stale=False)

    # volatility: persistence baseline - next hour looks like the last 4 hours
    if latest["rv_16"] is None:
        raise HTTPException(status_code=503, detail=f"rv_16 not available yet for {symbol} (feature warm-up)")
    return dict(symbol=symbol, task=task, open_time=latest["open_time"], horizon_candles=4,
                predicted_vol=latest["rv_16"], model_version="baseline-rv16", source="baseline", stale=False)


@app.get("/predict/{symbol}", response_model=Prediction, tags=["ml"])
def predict(
    symbol: str = PathParam(..., pattern=SYMBOL_PATTERN),
    task: Task = Query(Task.direction),
    interval: str = Query("15m", pattern=r"^\d{1,2}[mhd]$"),
    db: Database = Depends(get_db),
):
    """Newest prediction of the active model; naive baseline if no model is active yet."""
    return _predict(symbol, task, interval, db)


@app.get("/predict", response_model=Prediction, tags=["ml"])
def predict_query(
    symbol: str = Query("BTCUSDT", pattern=SYMBOL_PATTERN),
    task: Task = Query(Task.direction),
    interval: str = Query("15m", pattern=r"^\d{1,2}[mhd]$"),
    db: Database = Depends(get_db),
):
    """Same as /predict/{symbol}, with the symbol as a query parameter (default BTCUSDT)."""
    return _predict(symbol, task, interval, db)
