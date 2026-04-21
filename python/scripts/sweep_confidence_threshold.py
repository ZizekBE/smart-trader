"""ST-ALPHA-04 — confidence threshold sweep.

Runs HybridBacktestEngine across a grid of (core_min_conf, tact_min_conf)
values on 2024-01-01 → today data and ranks by Sharpe, trade count, win rate.

Usage::
    cd python && uv run python scripts/sweep_confidence_threshold.py
    cd python && uv run python scripts/sweep_confidence_threshold.py --days 365
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
    print(f"  ST-ALPHA-04  Confidence threshold sweep")
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

    # Grid: [core_conf] × [tact_conf]
    core_thresholds = [0.55, 0.60, 0.65, 0.70]
    tact_thresholds = [0.50, 0.55, 0.60, 0.65]

    results: list[dict] = []

    total = len(core_thresholds) * len(tact_thresholds)
    done  = 0
    for core_conf in core_thresholds:
        for tact_conf in tact_thresholds:
            done += 1
            print(f"  [{done:2d}/{total}]  core≥{core_conf:.2f}  tact≥{tact_conf:.2f}", end=" ", flush=True)
            try:
                eng = HybridBacktestEngine(
                    symbol=symbol,
                    initial_capital=capital,
                    core_min_conf=core_conf,
                    tact_min_conf=tact_conf,
                )
                r = eng.run(h1, h4, d1)
                cb = r.combined
                co = r.core.metrics
                ta = r.tactical.metrics
                results.append({
                    "core_conf": core_conf,
                    "tact_conf": tact_conf,
                    "trades":    cb.total_trades,
                    "win_rate":  round(cb.win_rate * 100, 1),
                    "sharpe":    round(cb.sharpe_ratio, 3),
                    "pnl":       round(cb.total_pnl, 2),
                    "max_dd":    round(cb.max_drawdown_pct, 2),
                    "core_trades": co.total_trades,
                    "tact_trades": ta.total_trades,
                })
                print(f"→ trades={cb.total_trades:3d}  WR={cb.win_rate*100:.0f}%  Sharpe={cb.sharpe_ratio:+.2f}  PnL=${cb.total_pnl:+.0f}")
            except Exception as e:
                print(f"→ ERROR: {e}")
                results.append({"core_conf": core_conf, "tact_conf": tact_conf, "error": str(e)})

    # ── Ranking ──────────────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r and r["trades"] >= 10]
    if not valid:
        print("\n  No valid results (need ≥10 trades). Check data coverage.")
        return

    df = pd.DataFrame(valid).sort_values("sharpe", ascending=False)

    print(f"\n{'─'*78}")
    print(f"  {'core≥':6s}  {'tact≥':6s}  {'trades':7s}  {'WR%':6s}  {'Sharpe':8s}  {'PnL':10s}  {'MaxDD%':7s}")
    print(f"  {'─'*70}")
    for _, row in df.head(8).iterrows():
        gate = "✓" if row["win_rate"] >= 40 and row["sharpe"] > 0 else " "
        print(f"  {row['core_conf']:.2f}     {row['tact_conf']:.2f}  "
              f"{int(row['trades']):7d}  {row['win_rate']:5.1f}%  "
              f"{row['sharpe']:+8.3f}  ${row['pnl']:+9.2f}  {row['max_dd']:6.2f}%  {gate}")

    print(f"\n  ✓ = gate pass (WR≥40% AND Sharpe>0)")

    best = df[df["win_rate"] >= 40].sort_values("sharpe", ascending=False)
    if not best.empty:
        b = best.iloc[0]
        print(f"\n  ── Recommended threshold ──────────────────────────────────")
        print(f"  core_min_conf = {b['core_conf']:.2f}   tact_min_conf = {b['tact_conf']:.2f}")
        print(f"  Sharpe={b['sharpe']:+.3f}  WR={b['win_rate']:.1f}%  trades={int(b['trades'])}  PnL=${b['pnl']:+.0f}")
    else:
        print("\n  No configuration meets WR≥40% gate. Review signal quality.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days",    type=int,   default=480,        help="Lookback days (default 480 ≈ 16 months)")
    p.add_argument("--symbol",  default="ETH/USDT",             help="Symbol to sweep")
    p.add_argument("--capital", type=float, default=10_000.0,   help="Initial capital")
    args = p.parse_args()
    asyncio.run(_run(args.days, args.symbol, args.capital))


if __name__ == "__main__":
    main()
