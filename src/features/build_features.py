"""GOLD: fact_kline (silver) -> gold_features.

Owner: Talgat  (Stage 2)

TODO:
- read clean candles per symbol/interval from fact_kline
- compute return_1, sma_20, rsi_14, volatility_20 and target_up
  (target_up = next candle's close > this candle's close)
- upsert into gold_features on (symbol_id, interval, open_time)

Run from the repo root:
    python src/features/build_features.py
"""
