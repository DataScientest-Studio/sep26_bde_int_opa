-- Latest feature row of one symbol (input for /features/latest and the baseline predictor).
-- Params: symbol, interval
SELECT s.symbol, g.interval, g.open_time, g.feature_set,
       g.ret_1, g.ret_4, g.ret_16, g.ret_96, g.ret_1_lag1, g.ret_1_lag2, g.ret_1_lag3,
       g.sma_20_ratio, g.sma_96_ratio, g.ema_12_ratio, g.ema_26_ratio, g.macd_rel, g.rsi_14,
       g.rv_16, g.rv_96, g.gk_var_16, g.atr_14_rel,
       g.hl_range_rel, g.body_rel, g.upper_wick_rel, g.lower_wick_rel,
       g.vol_z_96, g.vol_ratio_16, g.trades_z_96, g.log_quote_volume,
       g.hour_sin, g.hour_cos, g.dow_sin, g.dow_cos, g.min_to_funding,
       g.btc_ret_1, g.btc_ret_16
FROM gold_features g
JOIN dim_symbol s USING (symbol_id)
WHERE s.symbol = %(symbol)s
  AND g.interval = %(interval)s
ORDER BY g.open_time DESC
LIMIT 1;
