-- ============================================================
-- smart-trader: initial schema
-- ============================================================
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── OHLCV candles (TimescaleDB hypertable) ────────────────────
-- No PK here: TimescaleDB hypertables require unique constraints
-- to include the partition column.  Migration 002 adds the
-- UNIQUE (symbol, exchange, timeframe, time) constraint used for
-- upserts and SQLAlchemy identity tracking.
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    exchange    TEXT            NOT NULL,
    timeframe   TEXT            NOT NULL,  -- '1m','1h','1d','1w'
    open        NUMERIC(20, 8)  NOT NULL,
    high        NUMERIC(20, 8)  NOT NULL,
    low         NUMERIC(20, 8)  NOT NULL,
    close       NUMERIC(20, 8)  NOT NULL,
    volume      NUMERIC(30, 8)  NOT NULL
);

SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles (symbol, timeframe, time DESC);

-- ── Trading signals (regular table — UUID PK, lower volume) ──
CREATE TABLE IF NOT EXISTS signals (
    id              UUID            DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    symbol          TEXT            NOT NULL,
    timeframe       TEXT            NOT NULL,
    signal_type     TEXT            NOT NULL,  -- 'long','short','exit'
    source_model    TEXT            NOT NULL,  -- 'lgbm','transformer','ensemble'
    confidence      NUMERIC(5, 4)   NOT NULL,  -- 0.0000 – 1.0000
    market_regime   TEXT            NOT NULL,  -- 'bull','bear','ranging','volatile'
    strategy_mode   TEXT            NOT NULL,  -- 'conservative','balanced','aggressive'
    features        JSONB,                     -- raw feature snapshot
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, created_at DESC);

-- ── Trades (regular table — UUID PK) ─────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              UUID            DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_id       UUID            REFERENCES signals(id),
    opened_at       TIMESTAMPTZ     NOT NULL,
    closed_at       TIMESTAMPTZ,
    symbol          TEXT            NOT NULL,
    exchange        TEXT            NOT NULL,
    side            TEXT            NOT NULL,  -- 'buy','sell'
    entry_price     NUMERIC(20, 8)  NOT NULL,
    exit_price      NUMERIC(20, 8),
    quantity        NUMERIC(20, 8)  NOT NULL,
    pnl             NUMERIC(20, 8),
    pnl_pct         NUMERIC(10, 6),
    fees            NUMERIC(20, 8)  NOT NULL DEFAULT 0,
    slippage        NUMERIC(20, 8),
    status          TEXT            NOT NULL DEFAULT 'open',  -- 'open','closed','cancelled'
    exit_reason     TEXT,                      -- 'take_profit','stop_loss','signal','manual'
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

-- ── Signal labels (for model retraining) ─────────────────────
CREATE TABLE IF NOT EXISTS signal_labels (
    signal_id       UUID            REFERENCES signals(id) PRIMARY KEY,
    labeled_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    outcome         TEXT            NOT NULL,  -- 'good','bad','neutral'
    forward_return  NUMERIC(10, 6), -- actual return N bars after signal
    label_horizon   INT,            -- bars used for labeling
    alpha           NUMERIC(10, 6), -- return vs benchmark
    auto_labeled    BOOLEAN         NOT NULL DEFAULT TRUE
);

-- ── Model registry ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id              UUID            DEFAULT gen_random_uuid() PRIMARY KEY,
    registered_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    model_name      TEXT            NOT NULL,
    version         TEXT            NOT NULL,
    artifact_path   TEXT            NOT NULL,
    metrics         JSONB,          -- sharpe, accuracy, etc.
    is_active       BOOLEAN         NOT NULL DEFAULT FALSE,
    is_shadow       BOOLEAN         NOT NULL DEFAULT FALSE  -- A/B shadow mode
);

-- ── Daily trade stats (regular materialized view) ─────────────
-- trades is a regular table, so we use a standard materialized view
-- instead of a TimescaleDB continuous aggregate.
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_trade_stats AS
SELECT
    date_trunc('day', opened_at)    AS day,
    symbol,
    COUNT(*)                        AS trade_count,
    SUM(pnl)                        AS total_pnl,
    AVG(pnl_pct)                    AS avg_pnl_pct,
    SUM(fees)                       AS total_fees
FROM trades
WHERE status = 'closed'
GROUP BY date_trunc('day', opened_at), symbol
WITH NO DATA;
