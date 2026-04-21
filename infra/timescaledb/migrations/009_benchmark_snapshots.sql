-- Migration 009: Benchmark snapshots for EPIC-BENCH live tracking
--
-- benchmark_snapshots: one row per day per symbol.
--   portfolio_total = strategy equity at EOD
--   bh_price        = ETH/USDT spot at snapshot time
--   regime          = detected market regime
--   positions       = open position count
--
-- benchmark_baseline: one row per symbol — records day-0 price and
--   starting capital so B&H return can be computed at any time.

CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    ts              TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    portfolio_total NUMERIC(20, 8)  NOT NULL,
    cash            NUMERIC(20, 8)  NOT NULL,
    bh_price        NUMERIC(20, 8)  NOT NULL,
    regime          TEXT,
    positions       INT             NOT NULL DEFAULT 0,
    PRIMARY KEY (ts, symbol)
);

SELECT create_hypertable(
    'benchmark_snapshots', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

CREATE INDEX IF NOT EXISTS idx_bench_symbol_ts
    ON benchmark_snapshots (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS benchmark_baseline (
    symbol          TEXT            PRIMARY KEY,
    start_price     NUMERIC(20, 8)  NOT NULL,
    start_capital   NUMERIC(20, 8)  NOT NULL,
    start_ts        TIMESTAMPTZ     NOT NULL
);

COMMENT ON TABLE benchmark_snapshots IS
    'Daily EOD portfolio equity vs spot price — feeds benchmark_report.py';
COMMENT ON TABLE benchmark_baseline IS
    'Day-0 anchor for B&H return calculation per symbol';
