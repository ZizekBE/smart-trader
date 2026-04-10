#!/usr/bin/env python3
"""Bayesian walk-forward optimisation for the core (4h) sleeve.

Usage
─────
  uv run python scripts/run_core_optimization.py
  uv run python scripts/run_core_optimization.py --symbols BTC/USDT ETH/USDT --trials 150
  uv run python scripts/run_core_optimization.py --symbols BTC/USDT --days 365 --skip-sync

The script:
  1. Syncs latest 4h + 1d candles for each symbol
  2. Runs walk-forward Bayesian optimisation (Optuna TPE) on 4h signal TF
  3. Prints comparison table: default core params vs best params
  4. Saves results to the `optimization_runs` DB table (timeframe = "4h_core")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TF_LABEL = "4h_core"   # stored in optimization_runs.timeframe

# ── helpers ────────────────────────────────────────────────────────────────────

async def _load_df(symbol: str, tf: str, days: int) -> pd.DataFrame:
    from smart_trader.data.storage.candle_repo import CandleRepository
    from smart_trader.data.storage.database import get_session_factory

    factory = get_session_factory()
    now     = datetime.now(timezone.utc)
    since   = now - timedelta(days=days)

    async with factory() as session:
        repo    = CandleRepository(session)
        candles = await repo.get_range(symbol, "gateio", tf, since, now)

    if not candles:
        return pd.DataFrame()

    rows = [{"time": c.time, "open": float(c.open), "high": float(c.high),
             "low":  float(c.low),  "close": float(c.close), "volume": float(c.volume)}
            for c in candles]
    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


async def _sync(symbol: str, days: int) -> None:
    from smart_trader.core.settings import get_settings
    from smart_trader.data.ingestion.candle_service import CandleIngestionService
    from smart_trader.data.ingestion.gateio_client import GateIOClient

    s      = get_settings()
    client = GateIOClient(s.gateio_api_key.get_secret_value(),
                          s.gateio_api_secret.get_secret_value(), paper=True)
    svc    = CandleIngestionService(client)
    for tf, tf_days in [("4h", days), ("1d", days)]:
        try:
            n = await svc.sync(symbol, tf, limit_days=tf_days)
            log.info(f"  {symbol} {tf}: +{n} rows")
        except Exception as e:
            log.warning(f"  sync failed {symbol} {tf}: {e}")
    await client.close()


async def _save_result(symbol: str, result) -> None:  # type: ignore[type-arg]
    from sqlalchemy import text
    from smart_trader.data.storage.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text("UPDATE optimization_runs SET is_current=FALSE WHERE symbol=:s AND timeframe=:tf"),
            {"s": symbol, "tf": TF_LABEL},
        )
        for w in result.windows:
            await session.execute(text("""
                INSERT INTO optimization_runs
                  (symbol, timeframe, study_name, train_start, train_end, test_start, test_end,
                   n_trials, best_params, train_score, test_score,
                   train_sharpe, test_sharpe, train_pnl, test_pnl,
                   train_trades, test_trades, is_current)
                VALUES
                  (:symbol, :tf, :study, :trs, :tre, :tes, :tee,
                   :n, :params, :trscore, :tescore,
                   :trsh, :tesh, :trpnl, :tepnl,
                   :trtr, :tetr, TRUE)
            """), {
                "symbol":  symbol,      "tf":      TF_LABEL,
                "study":   result.study_name,
                "trs":     w.window.train_start, "tre": w.window.train_end,
                "tes":     w.window.test_start,  "tee": w.window.test_end,
                "n":       100,
                "params":  json.dumps(w.best_params),
                "trscore": w.train_score, "tescore": w.test_score,
                "trsh":    w.train_metrics.sharpe_ratio,
                "tesh":    w.test_metrics.sharpe_ratio,
                "trpnl":   w.train_metrics.total_pnl,
                "tepnl":   w.test_metrics.total_pnl,
                "trtr":    w.train_metrics.total_trades,
                "tetr":    w.test_metrics.total_trades,
            })
        await session.commit()


# ── reporting ─────────────────────────────────────────────────────────────────

_DEFAULT_CORE = {
    "min_confidence":        0.70,
    "atr_mult":              3.0,
    "rr_ratio":              3.0,
    "kelly_frac":            0.25,
    "trailing_atr_mult":     1.5,
    "trailing_trigger_pct":  0.75,
    "breakeven_trigger_pct": 0.50,
    "sl_cooldown_bars":      4,
    "min_conf_high_vol":     0.75,
}


def _print_comparison(symbol: str, default_m, opt_m, best_params: dict) -> None:  # type: ignore[type-arg]
    W = 60
    print(f"\n{'═'*W}")
    print(f"  {symbol}  —  core sleeve: default vs optimised (4h)")
    print(f"{'═'*W}")
    rows = [
        ("Trades",   str(default_m.total_trades),          str(opt_m.total_trades)),
        ("Win rate", f"{default_m.win_rate:.1f}%",         f"{opt_m.win_rate:.1f}%"),
        ("PnL",      f"${default_m.total_pnl:+.2f}",       f"${opt_m.total_pnl:+.2f}"),
        ("Sharpe",   f"{default_m.sharpe_ratio:.2f}",       f"{opt_m.sharpe_ratio:.2f}"),
        ("Max DD",   f"{default_m.max_drawdown_pct:.2f}%",  f"{opt_m.max_drawdown_pct:.2f}%"),
        ("Calmar",   f"{default_m.calmar_ratio:.2f}",       f"{opt_m.calmar_ratio:.2f}"),
    ]
    print(f"  {'Metric':<14} {'Default':>12} {'Optimised':>12}")
    print(f"  {'-'*38}")
    for k, d, o in rows:
        print(f"  {k:<14} {d:>12} {o:>12}")
    print(f"\n  Best params:")
    for k, v in best_params.items():
        default_val = _DEFAULT_CORE.get(k)
        delta = ""
        if isinstance(v, float) and default_val is not None:
            delta = f"  (was {default_val})"
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<28} {val}{delta}")

    print(f"\n  Suggested .env overrides:")
    print(f"    HYBRID_LONG_MIN_CONF={best_params.get('min_confidence', 0.70):.4f}")
    print(f"    HYBRID_LONG_ATR_MULT={best_params.get('atr_mult', 3.0):.4f}")
    print(f"    HYBRID_LONG_RR_RATIO={best_params.get('rr_ratio', 3.0):.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

async def _run(symbols: list[str], days: int, trials: int, skip_sync: bool) -> None:
    from smart_trader.analysis.backtest.engine import BacktestConfig, BacktestEngine
    from smart_trader.analysis.optimizer.core_sleeve_study import CoreSleeveOptimizer

    print(f"\n{'╔'+'═'*62+'╗'}")
    print(f"║   smart-trader · Core sleeve optimisation (4h){' '*15}║")
    print(f"║   symbols : {', '.join(symbols):<49}║")
    print(f"║   days    : {days:<49}║")
    print(f"║   trials  : {trials:<49}║")
    print(f"{'╚'+'═'*62+'╝'}\n")

    if not skip_sync:
        print("── syncing candles ──────────────────────────────────────────")
        for sym in symbols:
            await _sync(sym, days)

    for sym in symbols:
        print(f"\n── {sym} ─────────────────────────────────────────────────────")

        sig_df   = await _load_df(sym, "4h", days)
        trend_df = await _load_df(sym, "1d", days)

        if len(sig_df) < 60:
            log.warning(f"  Not enough 4h candles ({len(sig_df)}), skipping")
            continue

        log.info(f"  Loaded: 4h={len(sig_df)}, 1d={len(trend_df)}")

        # baseline with default core params
        default_cfg = BacktestConfig(
            symbol=sym, timeframe="4h", trend_timeframe="1d",
            require_mtf=False,
            min_confidence=_DEFAULT_CORE["min_confidence"],
            atr_mult=_DEFAULT_CORE["atr_mult"],
            rr_ratio=_DEFAULT_CORE["rr_ratio"],
        )
        default_m = BacktestEngine(default_cfg).run(sig_df, trend_df, None).metrics

        # optimise
        opt    = CoreSleeveOptimizer(
            symbol=sym,
            train_days=int(days * 0.67),
            test_days=int(days * 0.33),
            step_days=30,
        )
        result = opt.run(sig_df, trend_df, n_trials=trials)

        if not result.windows:
            log.warning("  No windows produced, skipping")
            continue

        best_cfg = opt._make_config(result.best_params)
        opt_m    = BacktestEngine(best_cfg).run(sig_df, trend_df, None).metrics

        _print_comparison(sym, default_m, opt_m, result.best_params)

        try:
            await _save_result(sym, result)
            log.info(f"  Saved to optimization_runs table (tf={TF_LABEL})")
        except Exception as e:
            log.warning(f"  DB save failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Core sleeve Bayesian optimisation")
    parser.add_argument("--symbols",   nargs="+", default=["BTC/USDT"])
    parser.add_argument("--days",      type=int, default=180)
    parser.add_argument("--trials",    type=int, default=100)
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args.symbols, args.days, args.trials, args.skip_sync))


if __name__ == "__main__":
    main()
