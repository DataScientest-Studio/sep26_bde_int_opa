# Architecture

## 1. Data flow (medallion architecture)

```mermaid
flowchart LR
    subgraph SRC["Source: Binance API"]
        REST["REST<br/>/api/v3/klines<br/>15m history"]
        WS["WebSocket<br/>&lt;symbol&gt;@kline_1m<br/>live candles"]
    end

    subgraph BRONZE["BRONZE · MongoDB"]
        RAW[("raw_klines<br/>raw JSON as received")]
    end

    subgraph SILVER["SILVER · PostgreSQL"]
        DIM[("dim_symbol")]
        FACT[("fact_kline<br/>clean candles")]
    end

    subgraph GOLD["GOLD · PostgreSQL"]
        FEAT[("gold_features<br/>outcomes")]
        PRED[("model_registry<br/>predictions")]
    end

    subgraph APP["Consumption (Stages 2–4)"]
        API["FastAPI"]
        UI["Streamlit"]
    end

    REST -- "extract_rest.py" --> RAW
    WS -- "extract_ws.py" --> RAW
    RAW -- "transform.py" --> FACT
    DIM --- FACT
    FACT -- "feature engineering<br/>(Stage 2)" --> FEAT
    FEAT -- "model · predict.py<br/>(Stage 2)" --> PRED
    FACT --> API
    PRED --> API
    API --> UI
```

| Layer | Storage | Content | Why this storage |
|---|---|---|---|
| Bronze | MongoDB `raw_klines` | Raw Binance responses, unchanged, plus `symbol`, `interval`, `source`, `ingested_at` | Schema-less: REST (array) and WebSocket (object) payloads have different shapes; keeping the original allows reprocessing without re-downloading |
| Silver | PostgreSQL `fact_kline`, `dim_symbol` | Typed, deduplicated candles in one structure for all symbols | Fixed schema, primary key prevents duplicates, joins and SQL aggregations |
| Gold | PostgreSQL `gold_features`, `outcomes`, `model_registry`, `predictions` | Model features, realized targets, trained models and their predictions | Ready-to-serve tables read by FastAPI and Streamlit |

## 2. Data model (UML class diagram)

```mermaid
classDiagram
    direction TB

    class raw_klines {
        <<MongoDB collection · BRONZE>>
        ObjectId _id
        string symbol
        string interval
        string source  "rest | websocket"
        datetime ingested_at
        array|object payload
    }

    class fact_kline {
        <<PostgreSQL · SILVER>>
        INT symbol_id PK, FK
        VARCHAR interval PK
        TIMESTAMPTZ open_time PK
        TIMESTAMPTZ close_time
        NUMERIC open
        NUMERIC high
        NUMERIC low
        NUMERIC close
        NUMERIC volume
        NUMERIC quote_volume
        INT trades
        VARCHAR source
        TIMESTAMPTZ loaded_at
    }

    class dim_symbol {
        <<PostgreSQL · SILVER>>
        SERIAL symbol_id PK
        VARCHAR symbol UNIQUE
        VARCHAR base_asset
        VARCHAR quote_asset
    }

    class gold_features {
        <<PostgreSQL · GOLD>>
        INT symbol_id PK, FK
        VARCHAR interval PK
        TIMESTAMPTZ open_time PK
        VARCHAR feature_set
        DOUBLE ret_1 .. ret_96, lags
        DOUBLE sma/ema ratios, macd_rel, rsi_14
        DOUBLE rv_16, rv_96, gk_var_16, atr_14_rel
        DOUBLE candle shape x4
        DOUBLE volume z-scores, log_quote_volume
        DOUBLE hour/dow sin-cos, min_to_funding
        DOUBLE btc_ret_1, btc_ret_16
    }

    class outcomes {
        <<PostgreSQL · GOLD>>
        INT symbol_id PK, FK
        VARCHAR interval PK
        TIMESTAMPTZ open_time PK
        SMALLINT horizon_candles PK
        DOUBLE realized_ret
        SMALLINT realized_dir
        DOUBLE realized_vol
        BOOLEAN realized_large
    }

    class model_registry {
        <<PostgreSQL · GOLD>>
        SERIAL model_id PK
        VARCHAR task
        INT symbol_id FK
        VARCHAR model_version
        VARCHAR algo
        JSONB feature_cols
        JSONB params
        JSONB metrics
        TEXT artifact_path
        BOOLEAN is_active
    }

    class predictions {
        <<PostgreSQL · GOLD>>
        BIGSERIAL prediction_id PK
        INT symbol_id FK
        VARCHAR interval
        TIMESTAMPTZ open_time
        VARCHAR task
        INT model_id FK
        SMALLINT horizon_candles
        SMALLINT y_pred_label
        DOUBLE y_pred_value
        DOUBLE y_baseline
        TIMESTAMPTZ predicted_at
    }

    raw_klines ..> fact_kline : transform.py
    fact_kline "*" <-- "1" dim_symbol
    fact_kline "1" --> "0..1" gold_features : build_features.py
    fact_kline "1" --> "0..2" outcomes : one row per horizon
    dim_symbol "1" --> "*" gold_features
    dim_symbol "1" --> "*" predictions
    model_registry "1" --> "*" predictions
    predictions ..> outcomes : scored by (open_time, horizon)
```

**How the data sets are linked**
- Bronze and silver are linked by the business key `(symbol, interval, open_time)`; `transform.py` maps `symbol` to `dim_symbol.symbol_id`.
- In Postgres, every table references `dim_symbol` through `symbol_id`, so adding a new trading pair is one row in `dim_symbol`.
- `fact_kline` and `gold_features` share the primary key `(symbol_id, interval, open_time)`: one feature row per candle.
- `outcomes` holds what really happened after each candle, per horizon (1 candle, 4 candles). It is built from `fact_kline` only, so it serves both as the training target and as the truth that predictions are scored against.
- `predictions` is one long table for all models: a row is `(candle, task, model_id)`. A new model is a new `task` value and a row in `model_registry`, not a new table. `model_registry` keeps metrics, parameters and the feature list of every trained model; exactly one model per task and symbol is `is_active`.
