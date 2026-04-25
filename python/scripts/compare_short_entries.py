"""ST-ALPHA-01 T-01-4/T-01-5 — long-only vs long+short backtest comparison.

Runs HybridBacktestEngine twice on the same data:
  A) long_only=True   — buy signals only (baseline)
  B) long_only=False  — buy + sell signals (short entries enabled)

Acceptance gate: B.Sharpe >= A.Sharpe − 0.2 (shorts must not degrade Sharpe).
Also reports long/short trade split per T-01-5.

Usage::
    cd python && uv run python scripts/compare_short_entries.py
    cd python && uv run python scripts/compare_short_entries.py --days 365
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
    print(f"  ST-ALPHA-01  Long-only vs Long+Short backtest")
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
    configs = [
        ("A  long-only",       True),
        ("B  long+short",      False),
    ]

    for label, long_only in configs:
        print(f"\n  Running {label}...", flush=True)
        try:
            eng = HybridBacktestEngine(
                symbol=symbol,
                initial_capital=capital,
                long_only=long_only,
            )
            r  = eng.run(h1, h4, d1)
            cb = r.combined
            n_long  = len(r.long_trades)
            n_short = len(r.short_trades)
            long_pnl  = sum(t.pnl for t in r.long_trades)
            short_pnl = sum(t.pnl for t in r.short_trades)
            results[label] = {
                "long_only": long_only,
                "trades":    cb.total_trades,
                "n_long":    n_long,
                "n_short":   n_short,
                "win_rate":  round(cb.win_rate * 100, 1),
                "sharpe":    round(cb.sharpe_ratio, 3),
                "pnl":       round(cb.total_pnl, 2),
                "long_pnl":  round(long_pnl, 2),
                "short_pnl": round(short_pnl, 2),
                "max_dd":    round(cb.max_drawdown_pct, 2),
            }
            print(f"    trades={cb.total_trades} (long={n_long}, short={n_short})  "
                  f"WR={cb.win_rate*100:.0f}%  Sharpe={cb.sharpe_ratio:+.3f}  "
                  f"PnL=${cb.total_pnl:+.0f}  MaxDD={cb.max_drawdown_pct:.2f}%")
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
    # Gate: shorts must not degrade Sharpe by more than 0.2
    gate_pass    = sharpe_delta >= -0.2

    print(f"\n{'─'*68}")
    print(f"  {'':44s}  {'A (long)':>10s}  {'B (both)':>10s}")
    print(f"  {'─'*66}")
    print(f"  {'Total trades':44s}  {a['trades']:>10d}  {b['trades']:>10d}")
    print(f"  {'  Long trades':44s}  {a['n_long']:>10d}  {b['n_long']:>10d}")
    print(f"  {'  Short trades':44s}  {a['n_short']:>10d}  {b['n_short']:>10d}")
    print(f"  {'Win rate':44s}  {a['win_rate']:>9.1f}%  {b['win_rate']:>9.1f}%")
    print(f"  {'Sharpe':44s}  {a['sharpe']:>+10.3f}  {b['sharpe']:>+10.3f}")
    print(f"  {'Total PnL':44s}  ${a['pnl']:>+9.2f}  ${b['pnl']:>+9.2f}")
    print(f"  {'  Long PnL':44s}  ${a['long_pnl']:>+9.2f}  ${b['long_pnl']:>+9.2f}")
    print(f"  {'  Short PnL':44s}  ${a['short_pnl']:>+9.2f}  ${b['short_pnl']:>+9.2f}")
    print(f"  {'Max drawdown':44s}  {a['max_dd']:>9.2f}%  {b['max_dd']:>9.2f}%")
    print(f"\n  Sharpe delta B-A: {sharpe_delta:+.3f}")
    print(f"  Gate (Δ Sharpe ≥ −0.2): {'✅ PASS — short entries accepted' if gate_pass else '❌ FAIL — shorts degrade Sharpe by more than 0.2'}")

    if gate_pass and b["n_short"] > 0:
        short_wr = sum(1 for t in [] if True)  # computed inline below
        print(f"\n  Short-only analysis:")
        print(f"    Short trade count : {b['n_short']}")
        print(f"    Short PnL         : ${b['short_pnl']:+.2f}")
        avg_short_pnl = b["short_pnl"] / b["n_short"] if b["n_short"] > 0 else 0
        print(f"    Avg short PnL     : ${avg_short_pnl:+.2f} per trade")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days",    type=int,   default=480)
    p.add_argument("--symbol",  default="ETH/USDT")
    p.add_argument("--capital", type=float, default=10_000.0)
    args = p.parse_args()
    asyncio.run(_run(args.days, args.symbol, args.capital))


if __name__ == "__main__":
    main()
