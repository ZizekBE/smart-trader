-- Feature Store tables for pre-computed indicators and agent state.
-- Supports the RL training pipeline and live inference cache.

-- Pre-computed features per candle (materialised at ingestion time)
CREATE TABLE IF NOT EXISTS computed_features (
    id          BIGSERIAL,
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    features    JSONB       NOT NULL,   -- {ema_9: 1.23, rsi: 45.6, ...}
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('computed_features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cf_sym_tf_time
    ON computed_features (symbol, timeframe, time DESC);

-- Agent decision log (for trade attribution and replay)
CREATE TABLE IF NOT EXISTS agent_decisions (
    id              BIGSERIAL,
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    regime          SMALLINT    NOT NULL,       -- 0-3
    target_position DOUBLE PRECISION NOT NULL,  -- [-1, 1]
    risk_budget     DOUBLE PRECISION NOT NULL,
    hold_bars       SMALLINT    NOT NULL,
    value_estimate  DOUBLE PRECISION NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL,
    model_version   TEXT        NOT NULL DEFAULT 'v0',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('agent_decisions', 'time', if_not_exists => TRUE);

-- Funding rate history (for futures RL training)
CREATE TABLE IF NOT EXISTS funding_rates (
    id              BIGSERIAL,
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    exchange        TEXT        NOT NULL,
    rate            DOUBLE PRECISION NOT NULL,
    next_funding    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fr_uniq
    ON funding_rates (symbol, exchange, time);

-- Open interest history
CREATE TABLE IF NOT EXISTS open_interest (
    id              BIGSERIAL,
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    exchange        TEXT        NOT NULL,
    value_usd       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('open_interest', 'time', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS idx_oi_uniq
    ON open_interest (symbol, exchange, time);

-- A/B test results
CREATE TABLE IF NOT EXISTS ab_test_results (
    id              SERIAL PRIMARY KEY,
    test_name       TEXT NOT NULL,
    champion_name   TEXT NOT NULL,
    challenger_name TEXT NOT NULL,
    status          TEXT NOT NULL,   -- running | champion_wins | challenger_wins | inconclusive
    champion_sharpe DOUBLE PRECISION,
    challenger_sharpe DOUBLE PRECISION,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
