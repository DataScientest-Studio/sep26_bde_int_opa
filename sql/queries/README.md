# SQL queries

Reusable queries used by the feature step, the API and the report.
One query per file, parameters as `%(name)s` (psycopg2 style).

| File | Used by | Purpose |
|---|---|---|
| `candles.sql` | API `/candles`, feature step | Clean candles of one symbol/interval, newest last |
| `stats.sql` | API `/stats` | Summary per symbol: count, period, min/max/avg price |
| `training_dataset.sql` | model training (Dennis) | Features + target from `gold_features`, ready for a DataFrame |
