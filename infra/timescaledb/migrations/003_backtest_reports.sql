-- ── Backtest reports — persist full results for replay ────────────────────
CREATE TABLE IF NOT EXISTS backtest_reports (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name             TEXT,                        -- optional user label
    symbol           TEXT        NOT NULL,
    timeframe        TEXT        NOT NULL,
    strategy_version TEXT        NOT NULL,
    days             INTEGER     NOT NULL,
    start_at         TIMESTAMPTZ,
    end_at           TIMESTAMPTZ,
    metrics          JSONB       NOT NULL,        -- PerformanceMetrics snapshot
    trades           JSONB       NOT NULL,        -- array of BacktestTrade dicts
    config           JSONB       NOT NULL         -- BacktestConfig snapshot
);

CREATE INDEX IF NOT EXISTS idx_backtest_reports_symbol
    ON backtest_reports (symbol, created_at DESC);
