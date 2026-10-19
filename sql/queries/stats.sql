-- Summary per symbol and interval for the /stats endpoint and the report.
SELECT s.symbol, f.interval,
       COUNT(*)             AS candles,
       MIN(f.open_time)     AS first_candle,
       MAX(f.open_time)     AS last_candle,
       MIN(f.low)           AS min_price,
       MAX(f.high)          AS max_price,
       ROUND(AVG(f.close), 2) AS avg_close,
       SUM(f.volume)        AS total_volume
FROM fact_kline f
JOIN dim_symbol s USING (symbol_id)
GROUP BY s.symbol, f.interval
ORDER BY s.symbol, f.interval;
