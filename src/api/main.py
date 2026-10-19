"""FastAPI application.

Owner: Talgat  (Stage 2)

Planned endpoints:
- GET /health            -> {"status": "ok"}
- GET /candles           -> clean candles (sql/queries/candles.sql)
- GET /stats             -> summary per symbol (sql/queries/stats.sql)
- GET /predict/{symbol}  -> calls the model's predict() (Dennis); stub until the model exists

Contract for predict(symbol: str) -> dict (implemented by Dennis in src/models/):
    {"symbol": "BTCUSDT", "open_time": "2026-10-19T09:00:00+00:00",
     "prob_up": 0.57, "predicted_up": True, "model_version": "v1"}

Run from the repo root:
    uvicorn src.api.main:app --reload
"""
