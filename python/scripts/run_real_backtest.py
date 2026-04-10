"""
Real-candle backtest — BTC / ETH / SOL, with optional v1 vs v2 comparison.

Usage:
    # 默认：v2，三个品种
    python scripts/run_real_backtest.py

    # 对比 v1 vs v2
    python scripts/run_real_backtest.py --versions v1 v2

    # 指定品种
    python scripts/run_real_backtest.py --symbols BTC/USDT ETH/USDT

    # 完整对比
    python scripts/run_real_backtest.py --symbols BTC/USDT ETH/USDT SOL/USDT --versions v1 v2
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_USER",     "trader")
os.environ.setdefault("DB_PASSWORD", "changeme")
os.environ.setdefault("DB_NAME",     "smart_trader")

from smart_trader.data.ingestion.gateio_client import GateIOClient
from smart_trader.data.ingestion.candle_service import CandleIngestionService
from smart_trader.analysis.analyzer import PerformanceAnalyzer
from smart_trader.analysis.reporting.report import PerformanceReport
from smart_trader.utils.logging import configure_logging

SYMBOLS         = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
SIGNAL_TF       = "1h"
MID_TF          = "4h"
TREND_TF        = "1d"
BACKFILL_DAYS   = 180
INITIAL_CAPITAL = 10_000.0
MIN_CONFIDENCE  = 0.65
MAX_POS_PCT     = 0.05


# ── candle sync ───────────────────────────────────────────────────────────────

async def ensure_candles(symbols: list[str]) -> None:
    client = GateIOClient(paper=True)
    svc    = CandleIngestionService(client)
    now    = datetime.now(timezone.utc)

    for symbol in symbols:
        for tf, days in [(SIGNAL_TF, BACKFILL_DAYS), (MID_TF, BACKFILL_DAYS * 3), (TREND_TF, 365)]:
            since = now - timedelta(days=days)
            print(f"  Syncing {symbol} {tf} ({days}d)…", end=" ", flush=True)
            try:
                n = await svc.backfill(symbol, tf, since=since, until=now)
                print(f"+{n} rows")
            except Exception as exc:
                print(f"FAILED: {exc}")

    await client.close()


# ── backtest ──────────────────────────────────────────────────────────────────

async def run_backtest(symbol: str, strategy_version: str) -> object:
    analyzer = PerformanceAnalyzer()
    now   = datetime.now(timezone.utc)
    since = now - timedelta(days=BACKFILL_DAYS)

    return await analyzer.backtest(
        symbol=symbol,
        timeframe=SIGNAL_TF,
        trend_timeframe=TREND_TF,
        mid_timeframe=MID_TF,
        initial_capital=INITIAL_CAPITAL,
        since=since,
        until=now,
        min_confidence=MIN_CONFIDENCE,
        max_position_pct=MAX_POS_PCT,
        sl_cooldown_bars=3,
        require_mtf=True,
        strategy_version=strategy_version,
    )


# ── display ───────────────────────────────────────────────────────────────────

def print_trades(trades) -> None:
    if not trades:
        print("  (no trades executed)")
        return

    print(f"\n  {'#':>3}  {'entry':>14}  {'exit':>14}  "
          f"{'side':>4}  {'entry $':>10}  {'exit $':>10}  "
          f"{'pnl':>8}  {'pnl%':>7}  {'reason':>14}  conf  regime")
    print("  " + "─" * 124)

    for i, t in enumerate(trades, 1):
        et   = t.entry_time.strftime("%m-%d %H:%M")
        xt   = t.exit_time.strftime("%m-%d %H:%M")
        sign = "+" if t.pnl >= 0 else ""
        print(
            f"  {i:>3}  {et:>14}  {xt:>14}  "
            f"{t.side:>4}  {t.entry_price:>10.2f}  {t.exit_price:>10.2f}  "
            f"{sign}{t.pnl:>7.2f}  {t.pnl_pct:>+7.2%}  {t.exit_reason:>14}  "
            f"{t.confidence:.2f}  {t.regime}"
        )


def _summary_row(symbol: str, version: str, result) -> tuple:
    m = result.metrics
    return (
        symbol, version,
        len(result.trades),
        f"{m.win_rate:.0%}",
        f"${m.total_pnl:+.2f}",
        f"{m.total_pnl_pct:+.2%}",
        f"{m.sharpe_ratio:.2f}",
        f"{m.max_drawdown_pct:.2%}",
    )


async def main() -> None:
    configure_logging("WARNING")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=SYMBOLS)
    parser.add_argument("--versions", nargs="+", default=["v2"],
                        help="Strategy versions to run (e.g. v1 v2)")
    args = parser.parse_args()

    symbols:  list[str] = args.symbols
    versions: list[str] = args.versions

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   smart-trader · real-candle backtest                       ║")
    print(f"║   symbols : {', '.join(symbols):<49}║")
    print(f"║   versions: {', '.join(versions):<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. sync candles once for all symbols
    print("\n── syncing candles ──────────────────────────────────────────────")
    await ensure_candles(symbols)

    # 2. run backtest per symbol × version
    summary_rows: list[tuple] = []

    for symbol in symbols:
        for version in versions:
            label = f"{symbol}  [{version}]"
            print(f"\n── {label}  {SIGNAL_TF}/{MID_TF}/{TREND_TF}  "
                  f"{BACKFILL_DAYS}d  capital=${INITIAL_CAPITAL:,.0f} ──")
            try:
                result = await run_backtest(symbol, version)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()
                continue

            report = PerformanceReport(
                result.metrics,
                title=f"Backtest  {label}  {SIGNAL_TF}  ({BACKFILL_DAYS}d)",
                start_at=result.start_at,
                end_at=result.end_at,
            )
            print(report.ascii())
            print(f"  TRADE LOG  ({len(result.trades)} trades)")
            print_trades(result.trades)

            summary_rows.append(_summary_row(symbol, version, result))

    # 3. comparison table
    if summary_rows:
        print("\n\n══ SUMMARY ════════════════════════════════════════════════════════════")
        print(f"  {'Symbol':<12}  {'Ver':>4}  {'Trades':>6}  {'WR':>5}  "
              f"{'PnL':>10}  {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
        print("  " + "─" * 74)

        # group by symbol to highlight v1→v2 delta
        prev: dict[str, tuple] = {}
        for row in summary_rows:
            sym, ver = row[0], row[1]
            print(f"  {sym:<12}  {ver:>4}  {row[2]:>6}  {row[3]:>5}  "
                  f"{row[4]:>10}  {row[5]:>8}  {row[6]:>7}  {row[7]:>7}")

            if len(versions) > 1 and sym in prev and ver != versions[0]:
                # print PnL delta between versions
                prev_pnl = float(prev[sym][4].replace("$", "").replace("+", ""))
                curr_pnl = float(row[4].replace("$", "").replace("+", ""))
                delta    = curr_pnl - prev_pnl
                sign     = "+" if delta >= 0 else ""
                print(f"  {'':12}  {'Δ':>4}  {'':>6}  {'':>5}  "
                      f"  {sign}{delta:>8.2f}")

            prev[sym] = row
        print()


if __name__ == "__main__":
    asyncio.run(main())
