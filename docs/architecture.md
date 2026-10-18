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
        FEAT[("gold_features")]
        PRED[("predictions")]
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
| Gold | PostgreSQL `gold_features`, `predictions` | Model features and predictions | Ready-to-serve tables read by FastAPI and Streamlit |

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
        NUMERIC return_1
        NUMERIC sma_20
        NUMERIC rsi_14
        NUMERIC volatility_20
        BOOLEAN target_up
    }

    class predictions {
        <<PostgreSQL · GOLD>>
        SERIAL prediction_id PK
        INT symbol_id FK
        VARCHAR interval
        TIMESTAMPTZ open_time
        VARCHAR model_version
        NUMERIC prob_up
        BOOLEAN predicted_up
        TIMESTAMPTZ created_at
    }

    raw_klines ..> fact_kline : transform.py
    fact_kline "*" <-- "1" dim_symbol
    fact_kline "1" --> "0..1" gold_features
    dim_symbol "1" --> "*" gold_features
    dim_symbol "1" --> "*" predictions
```

**How the data sets are linked**
- Bronze and silver are linked by the business key `(symbol, interval, open_time)`; `transform.py` maps `symbol` to `dim_symbol.symbol_id`.
- In Postgres, every table references `dim_symbol` through `symbol_id`, so adding a new trading pair is one row in `dim_symbol`.
- `fact_kline` and `gold_features` share the primary key `(symbol_id, interval, open_time)`: one feature row per candle.
- `predictions` keeps every model run (`model_version`, `created_at`), so predictions can be compared with the real outcome later.
