-- Clean candles for one symbol and interval, oldest first.
-- Params: symbol, interval, limit
SELECT s.symbol, f.interval, f.open_time, f.close_time,
       f.open, f.high, f.low, f.close, f.volume, f.quote_volume, f.trades
FROM fact_kline f
JOIN dim_symbol s USING (symbol_id)
WHERE s.symbol = %(symbol)s
  AND f.interval = %(interval)s
ORDER BY f.open_time DESC
LIMIT %(limit)s;
