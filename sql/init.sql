-- OPA: PostgreSQL schema (silver + gold layers)
-- Runs automatically on the first start of the postgres container.
-- To re-run after changes: docker compose down -v && docker compose up -d

-- ============ SILVER ============

CREATE TABLE IF NOT EXISTS dim_symbol (
    symbol_id    SERIAL PRIMARY KEY,
    symbol       VARCHAR(20) NOT NULL UNIQUE,
    base_asset   VARCHAR(10) NOT NULL,
    quote_asset  VARCHAR(10) NOT NULL
);

INSERT INTO dim_symbol (symbol, base_asset, quote_asset) VALUES
    ('BTCUSDT', 'BTC', 'USDT'),
    ('SOLUSDT', 'SOL', 'USDT')
ON CONFLICT (symbol) DO NOTHING;

CREATE TABLE IF NOT EXISTS fact_kline (
    symbol_id     INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval      VARCHAR(5)   NOT NULL,          -- '1m', '15m', ...
    open_time     TIMESTAMPTZ  NOT NULL,
    close_time    TIMESTAMPTZ  NOT NULL,
    open          NUMERIC(20, 8) NOT NULL,
    high          NUMERIC(20, 8) NOT NULL,
    low           NUMERIC(20, 8) NOT NULL,
    close         NUMERIC(20, 8) NOT NULL,
    volume        NUMERIC(28, 8) NOT NULL,        -- in base asset
    quote_volume  NUMERIC(28, 8) NOT NULL,        -- in USDT
    trades        INT          NOT NULL,
    source        VARCHAR(10)  NOT NULL,          -- 'rest' / 'websocket'
    loaded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, interval, open_time)  -- prevents duplicates on reruns
);

-- ============ GOLD (filled in Stage 2) ============

CREATE TABLE IF NOT EXISTS gold_features (
    symbol_id     INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval      VARCHAR(5)   NOT NULL,
    open_time     TIMESTAMPTZ  NOT NULL,
    return_1      NUMERIC,     -- % change vs previous candle
    sma_20        NUMERIC,
    rsi_14        NUMERIC,
    volatility_20 NUMERIC,
    target_up     BOOLEAN,     -- did the next candle close higher?
    PRIMARY KEY (symbol_id, interval, open_time)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  SERIAL PRIMARY KEY,
    symbol_id      INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval       VARCHAR(5)   NOT NULL,
    open_time      TIMESTAMPTZ  NOT NULL,     -- candle the prediction is for
    model_version  VARCHAR(50)  NOT NULL,
    prob_up        NUMERIC,
    predicted_up   BOOLEAN,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
