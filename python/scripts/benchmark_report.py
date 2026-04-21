"""EPIC-BENCH — strategy vs buy-and-hold benchmark report.

Usage::
    cd python && uv run python scripts/benchmark_report.py
    cd python && uv run python scripts/benchmark_report.py --days 30
    cd python && uv run python scripts/benchmark_report.py --symbol BTC/USDT --days 60
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


async def _run(symbol: str, days: int) -> None:
    from smart_trader.data.storage.benchmark_repo import BenchmarkRepository
    from smart_trader.data.storage.database import get_session_factory
    from smart_trader.data.storage.trade_repo import TradeRepository

    factory = get_session_factory()
    now     = datetime.now(timezone.utc)
    since   = now - timedelta(days=days)

    async with factory() as session:
        bench_repo = BenchmarkRepository(session)
        trade_repo = TradeRepository(session)

        baseline   = await bench_repo.get_baseline(symbol)
        snapshots  = await bench_repo.get_snapshots(symbol, since=since)
        trades     = await trade_repo.get_closed(symbol, since=since)

    if not snapshots:
        print(f"\n  No benchmark snapshots for {symbol} in the last {days} days.")
        print("  The loop must run at least one full day to record a snapshot.")
        return

    # ── B&H calculation ───────────────────────────────────────────────────────
    first = snapshots[0]
    last  = snapshots[-1]

    bh_entry   = float(first.bh_price)
    bh_current = float(last.bh_price)
    bh_return  = (bh_current - bh_entry) / bh_entry if bh_entry > 0 else 0.0

    start_cap  = float(baseline.start_capital) if baseline else float(first.portfolio_total)
    strat_return = (float(last.portfolio_total) - start_cap) / start_cap if start_cap > 0 else 0.0

    # ── trade stats ───────────────────────────────────────────────────────────
    n_trades  = len(trades)
    n_wins    = sum(1 for t in trades if float(t.pnl or 0) > 0)
    win_rate  = n_wins / n_trades if n_trades > 0 else 0.0
    total_pnl = sum(float(t.pnl or 0) for t in trades)

    # Sharpe from daily portfolio returns
    if len(snapshots) >= 2:
        import statistics
        daily_totals = [float(s.portfolio_total) for s in snapshots]
        daily_rets   = [(daily_totals[i] - daily_totals[i-1]) / daily_totals[i-1]
                        for i in range(1, len(daily_totals))]
        mean_r = statistics.mean(daily_rets) if daily_rets else 0.0
        std_r  = statistics.stdev(daily_rets) if len(daily_rets) > 1 else 0.0
        sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from snapshots
    peak   = start_cap
    max_dd = 0.0
    for s in snapshots:
        v    = float(s.portfolio_total)
        peak = max(peak, v)
        dd   = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # ── regime distribution ───────────────────────────────────────────────────
    regime_counts: dict[str, int] = {}
    for s in snapshots:
        r = s.regime or "unknown"
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total_snaps = len(snapshots)

    # ── output ────────────────────────────────────────────────────────────────
    start_date = snapshots[0].ts.strftime("%Y-%m-%d")
    end_date   = snapshots[-1].ts.strftime("%Y-%m-%d")
    n_days     = (snapshots[-1].ts - snapshots[0].ts).days + 1

    print(f"\n{'═'*68}")
    print(f"  EPIC-BENCH  Strategy vs Buy-and-Hold")
    print(f"  Symbol: {symbol}   Window: {start_date} → {end_date} ({n_days}d)")
    print(f"{'═'*68}\n")

    print(f"  {'':30s}  {'Strategy':>12s}  {'ETH B&H':>12s}")
    print(f"  {'─'*58}")
    print(f"  {'Return':30s}  {strat_return:+11.2%}  {bh_return:+11.2%}")
    print(f"  {'Portfolio value (end)':30s}  ${last.portfolio_total:>10,.2f}  "
          f"${start_cap * (1 + bh_return):>10,.2f}")
    print(f"  {'Sharpe (annualised)':30s}  {sharpe:>+11.3f}  {'—':>12s}")
    print(f"  {'Max drawdown':30s}  {max_dd:>11.2%}  {'—':>12s}")
    print(f"  {'Trades (closed)':30s}  {n_trades:>12d}  {'—':>12s}")
    print(f"  {'Win rate':30s}  {win_rate:>11.1%}  {'—':>12s}")
    print(f"  {'Realised PnL':30s}  ${total_pnl:>+10,.2f}  {'—':>12s}")

    # Edge vs B&H
    alpha = strat_return - bh_return
    gate_sharpe   = sharpe > 0
    gate_win_rate = win_rate >= 0.40 or n_trades == 0
    gate_drawdown = max_dd <= 0.15
    gate_days     = n_days >= 30
    all_pass      = gate_sharpe and gate_win_rate and gate_drawdown and gate_days

    print(f"\n  {'─'*58}")
    print(f"  Alpha vs B&H: {alpha:+.2%}")
    print(f"\n  30-Day Shadow Gate (configs/live_gate.json):")
    print(f"    Sharpe > 0       {'✓' if gate_sharpe else '✗'}  ({sharpe:+.3f})")
    print(f"    Win rate ≥ 40%   {'✓' if gate_win_rate else '✗'}  ({win_rate:.1%})")
    print(f"    Max DD ≤ 15%     {'✓' if gate_drawdown else '✗'}  ({max_dd:.2%})")
    print(f"    Days ≥ 30        {'✓' if gate_days else '✗'}  ({n_days}d)")
    print(f"\n  Gate result: {'✅ PASS — proceed to EPIC-PROD' if all_pass else '⏳ IN PROGRESS' if not gate_days else '❌ FAIL — return to EPIC-ALPHA'}")

    if regime_counts:
        print(f"\n  Regime distribution ({total_snaps} snapshots):")
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            pct = count / total_snaps * 100
            bar = "█" * int(pct / 5)
            print(f"    {regime:25s}  {pct:5.1f}%  {bar}")

    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETH/USDT")
    p.add_argument("--days",   type=int, default=30)
    args = p.parse_args()
    asyncio.run(_run(args.symbol, args.days))


if __name__ == "__main__":
    main()
