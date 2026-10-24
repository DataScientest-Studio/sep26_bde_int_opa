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

-- ============ GOLD (Stage 2) ============
-- Design: one wide feature table, one table with the realized outcomes (= training
-- targets), one long predictions table shared by all models, one model registry.
-- Adding a model is a new `task` value + a registry row, not a schema change.
-- All numbers are DOUBLE PRECISION: features feed ML, not accounting.

-- 1) Features: one row per closed candle. Every value uses only candles with
--    open_time <= this row's open_time (no look-ahead). Built by src/features/build_features.py
CREATE TABLE IF NOT EXISTS gold_features (
    symbol_id        INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval         VARCHAR(5)   NOT NULL,
    open_time        TIMESTAMPTZ  NOT NULL,
    feature_set      VARCHAR(10)  NOT NULL DEFAULT 'v1',   -- bump when a formula changes
    -- returns (log)
    ret_1            DOUBLE PRECISION,   -- log(close_t / close_t-1)
    ret_4            DOUBLE PRECISION,   -- 1 hour
    ret_16           DOUBLE PRECISION,   -- 4 hours
    ret_96           DOUBLE PRECISION,   -- 1 day
    ret_1_lag1       DOUBLE PRECISION,
    ret_1_lag2       DOUBLE PRECISION,
    ret_1_lag3       DOUBLE PRECISION,
    -- trend (relative to close, so BTC and SOL are comparable)
    sma_20_ratio     DOUBLE PRECISION,   -- close / SMA20 - 1
    sma_96_ratio     DOUBLE PRECISION,
    ema_12_ratio     DOUBLE PRECISION,
    ema_26_ratio     DOUBLE PRECISION,
    macd_rel         DOUBLE PRECISION,   -- (EMA12 - EMA26) / close
    rsi_14           DOUBLE PRECISION,
    -- volatility
    rv_16            DOUBLE PRECISION,   -- std of ret_1 over 16 candles
    rv_96            DOUBLE PRECISION,
    gk_var_16        DOUBLE PRECISION,   -- Garman-Klass variance, 16 candles
    atr_14_rel       DOUBLE PRECISION,   -- ATR14 / close
    -- candle shape
    hl_range_rel     DOUBLE PRECISION,   -- (high - low) / close
    body_rel         DOUBLE PRECISION,   -- (close - open) / (high - low)
    upper_wick_rel   DOUBLE PRECISION,
    lower_wick_rel   DOUBLE PRECISION,
    -- volume / activity
    vol_z_96         DOUBLE PRECISION,   -- (volume - mean96) / std96
    vol_ratio_16     DOUBLE PRECISION,   -- volume / mean16
    trades_z_96      DOUBLE PRECISION,
    log_quote_volume DOUBLE PRECISION,
    -- time of day / week (cyclic) and distance to funding (00/08/16 UTC)
    hour_sin         DOUBLE PRECISION,
    hour_cos         DOUBLE PRECISION,
    dow_sin          DOUBLE PRECISION,
    dow_cos          DOUBLE PRECISION,
    min_to_funding   DOUBLE PRECISION,
    -- cross-asset (BTC leads SOL); for BTCUSDT equals its own returns
    btc_ret_1        DOUBLE PRECISION,
    btc_ret_16       DOUBLE PRECISION,
    computed_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, interval, open_time)
);

-- 2) Outcomes: what really happened after candle t, per horizon (in candles).
--    Used as training targets AND to score predictions. Filled from fact_kline only,
--    so it is independent of any model. Horizons: 1 (next candle), 4 (next hour).
CREATE TABLE IF NOT EXISTS outcomes (
    symbol_id        INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval         VARCHAR(5)   NOT NULL,
    open_time        TIMESTAMPTZ  NOT NULL,
    horizon_candles  SMALLINT     NOT NULL,
    realized_ret     DOUBLE PRECISION NOT NULL,  -- log(close_t+h / close_t)
    realized_dir     SMALLINT     NOT NULL,       -- 1 if realized_ret > 0 else 0
    realized_vol     DOUBLE PRECISION,            -- std of ret_1 over t+1..t+h (NULL for h=1)
    realized_large   BOOLEAN      NOT NULL,       -- |realized_ret| > large_move_thr
    large_move_thr   DOUBLE PRECISION NOT NULL,   -- threshold used, e.g. 0.01
    filled_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, interval, open_time, horizon_candles)
);

-- 3) Model registry: one row per trained artifact. Exactly one active model per
--    (task, symbol) is enforced by the partial unique index below.
CREATE TABLE IF NOT EXISTS model_registry (
    model_id         SERIAL PRIMARY KEY,
    task             VARCHAR(30)  NOT NULL,   -- 'direction', 'volatility', 'large_move', 'regime', 'anomaly', 'baseline_*'
    symbol_id        INT          REFERENCES dim_symbol(symbol_id),  -- NULL = one model for all symbols
    interval         VARCHAR(5)   NOT NULL DEFAULT '15m',
    model_version    VARCHAR(60)  NOT NULL,   -- e.g. '2026-10-25T03:00Z-ab12cd' (train time + git sha)
    algo             VARCHAR(60)  NOT NULL,   -- 'sklearn.LogisticRegression', 'lightgbm.LGBMClassifier', ...
    feature_set      VARCHAR(10)  NOT NULL,
    feature_cols     JSONB        NOT NULL,   -- ordered list of gold_features columns used
    horizon_candles  SMALLINT     NOT NULL,   -- 0 for regime / anomaly
    params           JSONB        NOT NULL DEFAULT '{}',
    metrics          JSONB        NOT NULL DEFAULT '{}',  -- {"auc": 0.55, "baseline_auc": 0.50, ...}
    train_start      TIMESTAMPTZ  NOT NULL,
    train_end        TIMESTAMPTZ  NOT NULL,
    artifact_path    TEXT         NOT NULL,   -- models/<task>/<symbol>/<model_version>.joblib
    lib_versions     JSONB        NOT NULL DEFAULT '{}',
    trained_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    is_active        BOOLEAN      NOT NULL DEFAULT FALSE,
    promoted_at      TIMESTAMPTZ,
    UNIQUE (task, symbol_id, interval, model_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS model_registry_one_active
    ON model_registry (task, COALESCE(symbol_id, 0), interval) WHERE is_active;

-- 4) Predictions: one row per (candle, task, model). All models write here.
--    Scored later by joining outcomes on (symbol_id, interval, open_time, horizon_candles).
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    BIGSERIAL PRIMARY KEY,
    symbol_id        INT          NOT NULL REFERENCES dim_symbol(symbol_id),
    interval         VARCHAR(5)   NOT NULL,
    open_time        TIMESTAMPTZ  NOT NULL,   -- candle the features came from
    task             VARCHAR(30)  NOT NULL,
    model_id         INT          NOT NULL REFERENCES model_registry(model_id),
    horizon_candles  SMALLINT     NOT NULL,
    y_pred_label     SMALLINT,                -- class (0/1), regime state, anomaly flag
    y_pred_value     DOUBLE PRECISION,        -- P(up), predicted vol, anomaly score
    y_baseline       DOUBLE PRECISION,        -- naive forecast for the same row (0.5 / last vol)
    source           VARCHAR(10)  NOT NULL DEFAULT 'batch',  -- 'batch' (Airflow) | 'adhoc' (API)
    predicted_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (symbol_id, interval, open_time, task, model_id)
);
CREATE INDEX IF NOT EXISTS predictions_task_time
    ON predictions (task, symbol_id, open_time DESC);

-- 5) Reporting views (read by FastAPI / Streamlit, no joins needed there)
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (p.task, p.symbol_id, p.interval)
       s.symbol, p.interval, p.task, p.open_time, p.horizon_candles,
       p.y_pred_label, p.y_pred_value, p.y_baseline,
       r.model_version, p.predicted_at
FROM predictions p
JOIN model_registry r USING (model_id)
JOIN dim_symbol s ON s.symbol_id = p.symbol_id
WHERE r.is_active
ORDER BY p.task, p.symbol_id, p.interval, p.open_time DESC;

CREATE OR REPLACE VIEW v_prediction_quality AS
SELECT p.task, s.symbol, p.interval, r.model_version,
       date_trunc('day', p.open_time) AS day,
       COUNT(o.open_time)                                  AS n_scored,
       AVG((p.y_pred_label = o.realized_dir)::int)
           FILTER (WHERE p.task IN ('direction', 'baseline_direction', 'large_move')) AS hit_rate,
       SQRT(AVG(POWER(p.y_pred_value - o.realized_vol, 2))
           FILTER (WHERE p.task IN ('volatility', 'baseline_volatility'))) AS rmse,
       SQRT(AVG(POWER(p.y_baseline - o.realized_vol, 2))
           FILTER (WHERE p.task IN ('volatility', 'baseline_volatility'))) AS rmse_baseline
FROM predictions p
JOIN model_registry r USING (model_id)
JOIN dim_symbol s ON s.symbol_id = p.symbol_id
LEFT JOIN outcomes o
       ON o.symbol_id = p.symbol_id AND o.interval = p.interval
      AND o.open_time = p.open_time AND o.horizon_candles = p.horizon_candles
GROUP BY 1, 2, 3, 4, 5;
