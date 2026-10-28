-- Training dataset: features at candle t + what really happened after it.
-- One row per candle; rows without an outcome yet (the last `horizon` candles) are excluded.
-- Params: symbol, interval, horizon (candles: 1 = next candle, 4 = next hour)
--
-- Example (psycopg2):  cur.execute(open("sql/queries/training_dataset.sql").read(),
--                                  {"symbol": "BTCUSDT", "interval": "15m", "horizon": 1})
SELECT s.symbol, g.interval, g.open_time, g.feature_set,
       -- features
       g.ret_1, g.ret_4, g.ret_16, g.ret_96,
       g.ret_1_lag1, g.ret_1_lag2, g.ret_1_lag3,
       g.sma_20_ratio, g.sma_96_ratio, g.ema_12_ratio, g.ema_26_ratio, g.macd_rel, g.rsi_14,
       g.rv_16, g.rv_96, g.gk_var_16, g.atr_14_rel,
       g.hl_range_rel, g.body_rel, g.upper_wick_rel, g.lower_wick_rel,
       g.vol_z_96, g.vol_ratio_16, g.trades_z_96, g.log_quote_volume,
       g.hour_sin, g.hour_cos, g.dow_sin, g.dow_cos, g.min_to_funding,
       g.btc_ret_1, g.btc_ret_16,
       -- targets
       o.horizon_candles, o.realized_ret, o.realized_dir, o.realized_vol, o.realized_large
FROM gold_features g
JOIN dim_symbol s USING (symbol_id)
JOIN outcomes o
  ON o.symbol_id = g.symbol_id AND o.interval = g.interval AND o.open_time = g.open_time
WHERE s.symbol = %(symbol)s
  AND g.interval = %(interval)s
  AND o.horizon_candles = %(horizon)s
ORDER BY g.open_time;
