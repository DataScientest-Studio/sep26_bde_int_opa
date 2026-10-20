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

Detailed UML diagrams (data flow + data model) and design decisions: [docs/architecture.md](docs/architecture.md)

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
├── .env.example              # copy to .env (never commit .env)
├── requirements.txt
├── sql/
│   ├── init.sql              # Postgres schema (silver + gold)
│   └── queries/              # reusable SQL used by the API, features and report
├── src/
│   ├── data/
│   │   ├── db.py             # shared DB connections
│   │   ├── extract_rest.py   # bronze: history            (Dennis)
│   │   ├── extract_ws.py     # bronze: live               (Dennis)
│   │   ├── check_bronze.py   # bronze checks + samples    (Dennis)
│   │   └── transform.py      # silver: Mongo -> Postgres  (Talgat)
│   ├── features/
│   │   └── build_features.py # gold: fact_kline -> gold_features (Talgat, Stage 2)
│   ├── models/               # training notebook output, predict()  (Dennis, Stage 2)
│   ├── api/
│   │   └── main.py           # FastAPI                    (Talgat, Stage 2)
│   └── visualization/        # Streamlit / charts         (Stage 4)
├── models/                   # saved model artifacts (ignored by git)
├── notebooks/
├── tests/
├── docs/                     # UML, report, data samples
├── reports/figures/
└── scratch/                  # personal experiments, ignored by git (see scratch/README.md)
```

## Working rules

- **Shared code** lives in `src/`, `sql/`, `tests/`, `docs/` and goes through a pull request.
- **Personal experiments** go to `scratch/`: the folder is in `.gitignore`, so nothing there is ever committed.
  Need to test someone else's script with changes? Copy it to `scratch/` first, never edit their file.
- **More personal ignores** (a local tool, a data dump outside `scratch/`): add the path to `.git/info/exclude`.
  It works like `.gitignore` but stays on your machine and is never pushed.
- **Secrets and data** never go to git: `.env`, model files (`*.joblib`, `*.pkl`) and databases are ignored.
- **Contracts between our parts** are written down here or in the code: the Mongo document format (above),
  the `gold_features` columns (`sql/init.sql`) and the `predict()` output (`src/api/main.py`).

## Git workflow

- `master` always works; no direct pushes
- one branch per task: `feature/extract-rest`, `feature/postgres-schema`, ...
- small commits, clear messages
- pull request -> review by the other -> merge
- before a new task: `git checkout master && git pull`
