-- Migration 006: Portfolio state persistence
-- Tracks the all-time peak portfolio value so that drawdown calculation
-- survives process restarts.  Without this, a restart after drawdown
-- resets the peak to current value, making the circuit breaker ineffective.

CREATE TABLE IF NOT EXISTS portfolio_state (
    key         TEXT            PRIMARY KEY DEFAULT 'default',
    peak_value  NUMERIC(20, 8) NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE portfolio_state IS
    'Single-row key-value store for portfolio-wide state that must survive restarts';
