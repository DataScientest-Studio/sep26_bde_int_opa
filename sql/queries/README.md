# SQL queries

Reusable queries used by the feature step, the API and the report.
One query per file, parameters as `%(name)s` (psycopg2 style).

| File | Used by | Purpose |
|---|---|---|
| `candles.sql` | API `/candles` | Clean candles of one symbol/interval, newest first |
| `stats.sql` | API `/stats` | Summary per symbol: count, period, min/max/avg price |
| `training_dataset.sql` | model training (Dennis) | Features (`gold_features`) joined with targets (`outcomes`); params `symbol, interval, horizon` |
| `latest_features.sql` | API `/features/latest` | Newest feature row of one symbol |
| `latest_prediction.sql` | API `/predict` | Newest prediction of the active model (`v_latest_predictions`) |
| `baseline_direction.sql` | API `/predict` (fallback) | Naive P(up) = share of up-candles in the last day |
| `symbols.sql` | API `/symbols` | Trading pairs from `dim_symbol` |
