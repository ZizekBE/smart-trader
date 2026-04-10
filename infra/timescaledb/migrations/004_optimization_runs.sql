-- Bayesian walk-forward optimization results
CREATE TABLE IF NOT EXISTS optimization_runs (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL    DEFAULT NOW(),
    symbol       TEXT        NOT NULL,
    timeframe    TEXT        NOT NULL,
    study_name   TEXT        NOT NULL,
    train_start  TIMESTAMPTZ NOT NULL,
    train_end    TIMESTAMPTZ NOT NULL,
    test_start   TIMESTAMPTZ NOT NULL,
    test_end     TIMESTAMPTZ NOT NULL,
    n_trials     INTEGER     NOT NULL,
    best_params  JSONB       NOT NULL,
    train_score  FLOAT,       -- sharpe × (1 − maxdd)
    test_score   FLOAT,
    train_sharpe FLOAT,
    test_sharpe  FLOAT,
    train_pnl    FLOAT,
    test_pnl     FLOAT,
    train_trades INTEGER,
    test_trades  INTEGER,
    is_current   BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_opt_runs_symbol_tf
    ON optimization_runs (symbol, timeframe, created_at DESC);
