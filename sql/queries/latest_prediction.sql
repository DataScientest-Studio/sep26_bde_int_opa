-- Latest prediction of the ACTIVE model for one symbol and task (written by the ML pipeline).
-- Params: symbol, interval, task
SELECT symbol, interval, task, open_time, horizon_candles,
       y_pred_label, y_pred_value, y_baseline, model_version, predicted_at
FROM v_latest_predictions
WHERE symbol = %(symbol)s
  AND interval = %(interval)s
  AND task = %(task)s;
