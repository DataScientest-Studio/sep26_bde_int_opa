# OPA – Crypto Data Pipeline (Binance)

DataScientest Data Engineering project (sep26_bde_int_opa).

## Architecture

```
Binance REST (history) ─┐
                        ├─> MongoDB: raw_klines   (BRONZE, raw data as received)
Binance WebSocket (live)┘          │
                                   ▼  transform.py
                         PostgreSQL: fact_kline    (SILVER, clean candles)
                                   │
                                   ▼
                         PostgreSQL: gold_features, predictions (GOLD)
                                   │
                                   ▼
                         FastAPI ──> Streamlit
```

- **Assets:** BTCUSDT, SOLUSDT
- **REST:** 15m candles, ~6 months history
- **WebSocket:** 1m live candles

## Local setup

Requirements: Docker, Python 3.10+

```bash
git clone https://github.com/DataScientest-Studio/sep26_bde_int_opa.git
cd sep26_bde_int_opa
cp .env.example .env
docker compose up -d                 # starts Postgres + Mongo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/data/db.py                # checks that both databases are reachable
```

Each of us runs the databases locally. Data is never committed to git.

## Mongo document format (bronze contract)

Collection: `raw_klines`. Every extractor writes documents in this format:

| Field | Type | Example |
|---|---|---|
| `symbol` | string | `"BTCUSDT"` |
| `interval` | string | `"15m"` |
| `source` | string | `"rest"` or `"websocket"` |
| `ingested_at` | datetime (UTC) | `2026-10-16T13:00:00Z` |
| `payload` | array / object | raw kline as received (array from REST, object from WebSocket) |

Do not change this format without telling the team first.

## Repo structure

Based on the DataScientest template (cookiecutter data science).

```
├── docker-compose.yml        # Postgres + Mongo
├── .env.example              # copy to .env
├── requirements.txt
├── sql/init.sql              # Postgres schema (silver + gold)
├── src/
│   ├── data/
│   │   ├── db.py             # shared DB connections
│   │   ├── extract_rest.py   # bronze: history  (Dennis)
│   │   ├── extract_ws.py     # bronze: live     (Dennis)
│   │   └── transform.py      # silver: Mongo -> Postgres (Talgat)
│   ├── features/             # gold: feature engineering (Stage 2)
│   ├── models/               # train / predict (Stage 2)
│   └── visualization/
├── notebooks/
├── tests/
├── docs/                     # UML, report
└── reports/figures/
```

## Git workflow

- `master` always works; no direct pushes
- one branch per task: `feature/extract-rest`, `feature/postgres-schema`, ...
- small commits, clear messages
- pull request -> review by the other -> merge
- before a new task: `git checkout master && git pull`
