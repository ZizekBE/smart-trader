#!/usr/bin/env python3
"""Multi-symbol hybrid backtest.

Usage
─────
  uv run python scripts/run_hybrid_backtest.py
  uv run python scripts/run_hybrid_backtest.py --symbols BTC/USDT ETH/USDT SOL/USDT
  uv run python scripts/run_hybrid_backtest.py --days 365 --skip-sync
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


async def _load_core_params(symbol: str) -> dict | None:
    """Load the current optimised core-sleeve params for a symbol from the DB."""
    from sqlalchemy import text
    from smart_trader.data.storage.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        row = await session.execute(
            text("""SELECT best_params FROM optimization_runs
                    WHERE symbol=:s AND timeframe='4h_core' AND is_current=TRUE
                    ORDER BY id DESC LIMIT 1"""),
            {"s": symbol},
        )
        r = row.fetchone()
        return r.best_params if r else None  # JSONB comes back as dict


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
    for tf in ("1h", "4h", "1d"):
        try:
            n = await svc.sync(symbol, tf, limit_days=days)
            log.info(f"  {symbol} {tf}: +{n} rows")
        except Exception as e:
            log.warning(f"  sync failed {symbol} {tf}: {e}")
    await client.close()


def _print_result(sym: str, result) -> None:  # type: ignore[type-arg]
    co, ta, cb = result.core, result.tactical, result.combined
    W = 62
    print(f"\n{'─'*W}")
    print(f"  {sym}")
    print(f"{'─'*W}")
    print(f"  {'Sleeve':<12} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'-'*56}")

    def _row(label, m):
        return (f"  {label:<12} {m.total_trades:>7} "
                f"{m.win_rate*100:>6.1f}% {m.total_pnl:>+10.2f} "
                f"{m.sharpe_ratio:>8.2f} {m.max_drawdown_pct:>7.2f}%")

    print(_row("Core (4h)",     co.metrics))
    print(_row("Tactical (1h)", ta.metrics))
    print(f"  {'─'*56}")
    print(_row("Combined",      cb))


async def _run(symbols: list[str], days: int, skip_sync: bool, capital: float) -> None:
    from smart_trader.analysis.backtest.hybrid_engine import HybridBacktestEngine
    from smart_trader.analysis.metrics.calculator import MetricsCalculator, TradeResult

    print(f"\n{'╔'+'═'*62+'╗'}")
    print(f"║   smart-trader · Hybrid multi-symbol backtest{' '*16}║")
    print(f"║   symbols : {', '.join(symbols):<49}║")
    print(f"║   days    : {days:<49}║")
    print(f"║   capital : ${capital:,.0f}{' '*(48-len(f'{capital:,.0f}'))}║")
    print(f"{'╚'+'═'*62+'╝'}")

    if not skip_sync:
        print("\n── syncing candles ──────────────────────────────────────────")
        for sym in symbols:
            await _sync(sym, days)

    all_sym_results = []

    for sym in symbols:
        print(f"\n── {sym} ──")
        h1 = await _load_df(sym, "1h", days)
        h4 = await _load_df(sym, "4h", days)
        d1 = await _load_df(sym, "1d", days)

        if len(h1) < 120:
            log.warning(f"  Not enough 1h candles ({len(h1)}), skipping")
            continue

        log.info(f"  Loaded: 1h={len(h1)}, 4h={len(h4)}, 1d={len(d1)}")

        core_params = await _load_core_params(sym)
        if core_params:
            log.info(f"  Core params (optimised): min_conf={core_params.get('min_confidence', '?'):.3f}  "
                     f"atr={core_params.get('atr_mult', '?'):.3f}  rr={core_params.get('rr_ratio', '?'):.3f}")
        else:
            log.info(f"  Core params: using global settings (no optimised run found)")

        result = HybridBacktestEngine(symbol=sym, initial_capital=capital, core_params=core_params).run(h1, h4, d1)
        _print_result(sym, result)
        all_sym_results.append((sym, result))

    if len(all_sym_results) > 1:
        # aggregate: treat each symbol's PnL as contributing to a shared portfolio
        all_trades = []
        for _sym, r in all_sym_results:
            all_trades.extend(r.all_trades)
        all_trades.sort(key=lambda t: t.entry_time)

        trade_results = [
            TradeResult(
                pnl=t.pnl, pnl_pct=t.pnl_pct,
                opened_at=t.entry_time, closed_at=t.exit_time,
                fees=t.fees, slippage=t.slippage, side=t.side,
            )
            for t in all_trades
        ]
        agg = MetricsCalculator().calculate(trade_results, capital * len(all_sym_results))

        W = 62
        print(f"\n{'═'*W}")
        print(f"  PORTFOLIO AGGREGATE  ({len(all_sym_results)} symbols)")
        print(f"{'═'*W}")
        print(f"  Trades   : {agg.total_trades}")
        print(f"  Win rate : {agg.win_rate*100:.1f}%")
        print(f"  PnL      : ${agg.total_pnl:+.2f}")
        print(f"  Sharpe   : {agg.sharpe_ratio:.2f}")
        print(f"  Max DD   : {agg.max_drawdown_pct:.2f}%")
        print(f"  Calmar   : {agg.calmar_ratio:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-symbol hybrid backtest")
    parser.add_argument("--symbols",   nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    parser.add_argument("--days",      type=int,   default=180)
    parser.add_argument("--capital",   type=float, default=10_000.0)
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args.symbols, args.days, args.skip_sync, args.capital))


if __name__ == "__main__":
    main()
