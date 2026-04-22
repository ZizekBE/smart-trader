"""EPIC-PHASE2 T-01-4 — regime routing vs single-strategy comparison.

Runs HybridBacktestEngine twice on the same data:
  A) regime_routing=False  — full v2 strategy for all regimes
  B) regime_routing=True   — trend_follower / mean_reversion / breakout by regime

Accepts result if B.Sharpe >= A.Sharpe + 0.3.

Usage::
    cd python && uv run python scripts/compare_regime_routing.py
    cd python && uv run python scripts/compare_regime_routing.py --days 365
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


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


async def _run(days: int, symbol: str, capital: float) -> None:
    from smart_trader.analysis.backtest.hybrid_engine import HybridBacktestEngine

    print(f"\n{'═'*68}")
    print(f"  EPIC-PHASE2  Regime routing vs single-strategy")
    print(f"  Symbol: {symbol}   Days: {days}   Capital: ${capital:,.0f}")
    print(f"{'═'*68}\n")

    print("  Loading candles...", end=" ", flush=True)
    h1 = await _load_df(symbol, "1h", days)
    h4 = await _load_df(symbol, "4h", days)
    d1 = await _load_df(symbol, "1d", days)
    print(f"1h={len(h1)}, 4h={len(h4)}, 1d={len(d1)}")

    if len(h1) < 200 or len(h4) < 50:
        print("  ERROR: Not enough candles. Run backfill first.")
        return

    results = {}
    for label, routing in [("A  single-strategy (v2 all detectors)", False),
                            ("B  regime-routing  (trend/mean-rev/breakout)", True)]:
        print(f"\n  Running {label}...", flush=True)
        try:
            eng = HybridBacktestEngine(
                symbol=symbol,
                initial_capital=capital,
                regime_routing=routing,
            )
            r  = eng.run(h1, h4, d1)
            cb = r.combined
            results[label] = {
                "routing": routing,
                "trades":   cb.total_trades,
                "win_rate": round(cb.win_rate * 100, 1),
                "sharpe":   round(cb.sharpe_ratio, 3),
                "pnl":      round(cb.total_pnl, 2),
                "max_dd":   round(cb.max_drawdown_pct, 2),
            }
            print(f"    trades={cb.total_trades}  WR={cb.win_rate*100:.0f}%  "
                  f"Sharpe={cb.sharpe_ratio:+.3f}  PnL=${cb.total_pnl:+.0f}  "
                  f"MaxDD={cb.max_drawdown_pct:.2f}%")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[label] = {"error": str(e)}

    # ── comparison table ─────────────────────────────────────────────────────
    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) < 2:
        print("\n  Cannot compare — one or both runs failed.")
        return

    vals = list(valid.values())
    a, b = vals[0], vals[1]
    sharpe_delta = b["sharpe"] - a["sharpe"]
    gate_pass    = sharpe_delta >= 0.3

    print(f"\n{'─'*68}")
    print(f"  {'':40s}  {'A (base)':>10s}  {'B (routed)':>10s}")
    print(f"  {'─'*62}")
    print(f"  {'Trades':40s}  {a['trades']:>10d}  {b['trades']:>10d}")
    print(f"  {'Win rate':40s}  {a['win_rate']:>9.1f}%  {b['win_rate']:>9.1f}%")
    print(f"  {'Sharpe':40s}  {a['sharpe']:>+10.3f}  {b['sharpe']:>+10.3f}")
    print(f"  {'Total PnL':40s}  ${a['pnl']:>+9.2f}  ${b['pnl']:>+9.2f}")
    print(f"  {'Max drawdown':40s}  {a['max_dd']:>9.2f}%  {b['max_dd']:>9.2f}%")
    print(f"\n  Sharpe delta B-A: {sharpe_delta:+.3f}")
    print(f"  Gate (Δ Sharpe ≥ 0.3): {'✅ PASS — keep regime routing' if gate_pass else '❌ FAIL — routing does not improve Sharpe'}")

    if not gate_pass and sharpe_delta > 0:
        print(f"  Note: routing is positive (+{sharpe_delta:.3f}) but below 0.3 gate. "
              "Keep routing ON — it doesn't hurt, and live data will decide.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days",    type=int,   default=480)
    p.add_argument("--symbol",  default="ETH/USDT")
    p.add_argument("--capital", type=float, default=10_000.0)
    args = p.parse_args()
    asyncio.run(_run(args.days, args.symbol, args.capital))


if __name__ == "__main__":
    main()
