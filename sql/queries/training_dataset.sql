-- Training dataset for the model: features + target, one row per candle.
-- Params: symbol, interval
SELECT s.symbol, g.interval, g.open_time,
       g.return_1, g.sma_20, g.rsi_14, g.volatility_20,
       g.target_up
FROM gold_features g
JOIN dim_symbol s USING (symbol_id)
WHERE s.symbol = %(symbol)s
  AND g.interval = %(interval)s
  AND g.target_up IS NOT NULL          -- last candle has no "next candle" yet
ORDER BY g.open_time;
