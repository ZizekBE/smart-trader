-- Migration 005: Hybrid strategy sleeve support
-- Adds `sleeve` column to trades table so each trade is tagged
-- as belonging to the 'core' (long-term) or 'tactical' (short-term) sleeve.

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS sleeve TEXT NOT NULL DEFAULT 'tactical';

COMMENT ON COLUMN trades.sleeve IS
    'Which strategy sleeve opened this trade: core (long-term) | tactical (short-term)';

CREATE INDEX IF NOT EXISTS idx_trades_sleeve
    ON trades (sleeve);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_sleeve_status
    ON trades (symbol, sleeve, status);
