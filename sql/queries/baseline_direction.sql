-- Naive direction baseline: share of up-candles over the last 96 candles (= 1 day of 15m),
-- used as P(next candle up) until a trained model is active.
-- Params: symbol, interval
SELECT AVG(realized_dir)::float AS prob_up, COUNT(*) AS n
FROM (
    SELECT o.realized_dir
    FROM outcomes o
    JOIN dim_symbol s USING (symbol_id)
    WHERE s.symbol = %(symbol)s
      AND o.interval = %(interval)s
      AND o.horizon_candles = 1
    ORDER BY o.open_time DESC
    LIMIT 96
) last_day;
