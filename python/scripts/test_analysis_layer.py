"""
Analysis Layer integration test.

Layers tested:
  1. MetricsCalculator  — pure-Python metrics from synthetic trade list
  2. PerformanceReport  — ASCII + JSON rendering
  3. BacktestEngine     — in-memory replay on synthetic OHLCV candles
  4. PerformanceAnalyzer (live) — reads real closed trades from DB

Run:
    python scripts/test_analysis_layer.py
"""
from __future__ import annotations

import asyncio
import math
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

import pandas as pd
import numpy as np

from smart_trader.analysis.metrics.calculator import MetricsCalculator, TradeResult
from smart_trader.analysis.reporting.report import PerformanceReport
from smart_trader.analysis.backtest.engine import BacktestConfig, BacktestEngine
from smart_trader.analysis.analyzer import PerformanceAnalyzer
from smart_trader.utils.logging import configure_logging


def ok(msg):    print(f"  ✓  {msg}")
def fail(msg):  print(f"  ✗  {msg}"); sys.exit(1)
def section(t): print(f"\n{'─'*56}\n  {t}\n{'─'*56}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_trades(n: int = 20, win_rate: float = 0.60) -> list[TradeResult]:
    """Generate synthetic closed trades."""
    rng   = np.random.default_rng(42)
    now   = datetime.now(timezone.utc)
    trades = []
    for i in range(n):
        opened = now - timedelta(hours=int(n - i + rng.integers(1, 4)))
        closed = opened + timedelta(hours=int(rng.integers(1, 24)))
        win    = rng.random() < win_rate
        pnl    = float(rng.uniform(50, 200)) if win else float(-rng.uniform(20, 80))
        trades.append(TradeResult(
            pnl=pnl,
            pnl_pct=pnl / 500,
            opened_at=opened,
            closed_at=closed,
            fees=abs(pnl) * 0.002,
            slippage=abs(pnl) * 0.001,
            side="buy" if win else "sell",
        ))
    return trades


def _make_ohlcv(n: int = 300, start_price: float = 85_000.0) -> pd.DataFrame:
    """Synthetic BTC-like hourly OHLCV (random walk with slight upward drift)."""
    rng = np.random.default_rng(7)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    times  = [now - timedelta(hours=n - i) for i in range(n)]
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.normal(0.0002, 0.008)))

    rows = []
    for i, (t, c) in enumerate(zip(times, closes)):
        spread = c * 0.003
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) + rng.uniform(0, spread)
        l = min(o, c) - rng.uniform(0, spread)
        v = float(rng.uniform(100, 500))
        rows.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})

    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ── test 1: MetricsCalculator ─────────────────────────────────────────────────

def test_metrics_calculator():
    section("1 · MetricsCalculator — pure metrics")
    trades  = _make_trades(40, win_rate=0.60)
    calc    = MetricsCalculator()
    metrics = calc.calculate(trades, initial_capital=10_000.0)

    ok(f"total_trades={metrics.total_trades}  win_rate={metrics.win_rate:.1%}")
    ok(f"total_pnl=${metrics.total_pnl:.2f}  profit_factor={metrics.profit_factor:.3f}")
    ok(f"sharpe={metrics.sharpe_ratio:.4f}  sortino={metrics.sortino_ratio:.4f}")
    ok(f"max_drawdown_pct={metrics.max_drawdown_pct:.2%}  max_drawdown_usd=${metrics.max_drawdown_usd:.2f}")
    ok(f"total_fees=${metrics.total_fees:.4f}  fee_drag={metrics.fee_drag_pct:.4%}")
    ok(f"avg_duration={metrics.avg_trade_duration_hours:.2f} h")

    assert metrics.total_trades == 40
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.gross_profit >= 0
    assert metrics.gross_loss >= 0
    assert metrics.profit_factor > 0
    assert 0.0 <= metrics.max_drawdown_pct <= 1.0
    assert metrics.final_capital == metrics.initial_capital + metrics.total_pnl

    # empty trades → zero metrics
    empty = calc.calculate([], initial_capital=5_000.0)
    assert empty.total_trades == 0
    assert empty.initial_capital == 5_000.0
    assert empty.final_capital == 5_000.0
    ok("empty trades → zero-filled PerformanceMetrics ✓")


# ── test 2: PerformanceReport ─────────────────────────────────────────────────

def test_report():
    section("2 · PerformanceReport — ASCII + JSON")
    trades  = _make_trades(20)
    metrics = MetricsCalculator().calculate(trades, 10_000.0)
    report  = PerformanceReport(metrics, title="Synthetic Test Report")

    ascii_out = report.ascii()
    ok(f"ASCII report length={len(ascii_out)} chars")
    assert "Win rate" in ascii_out
    assert "Sharpe" in ascii_out
    assert "CAPITAL" in ascii_out

    d = report.to_dict()
    assert "pnl" in d
    assert "risk" in d
    ok("JSON dict keys: " + ", ".join(d.keys()))

    json_str = report.to_json()
    import json
    parsed = json.loads(json_str)
    assert parsed["overview"]["total_trades"] == 20
    ok("JSON round-trip verified")

    print(ascii_out)


# ── test 3: BacktestEngine ────────────────────────────────────────────────────

def test_backtest_engine():
    section("3 · BacktestEngine — in-memory replay")

    df = _make_ohlcv(n=300)
    ok(f"Candle window: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)")

    cfg    = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_capital=10_000.0,
        min_confidence=0.50,   # lower threshold → more trades in synthetic data
    )
    engine = BacktestEngine(cfg)
    result = engine.run(signal_candles=df, trend_candles=df)

    m = result.metrics
    ok(f"Backtest trades={m.total_trades}  win_rate={m.win_rate:.1%}  pnl=${m.total_pnl:.2f}")
    ok(f"Sharpe={m.sharpe_ratio:.4f}  Sortino={m.sortino_ratio:.4f}  Calmar={m.calmar_ratio:.4f}")
    ok(f"Max drawdown={m.max_drawdown_pct:.2%}")

    assert isinstance(result.trades, list)
    assert m.total_trades == len(result.trades)
    assert m.initial_capital == 10_000.0
    # all closed trades have valid exit reasons
    valid_reasons = {"stop_loss", "take_profit", "end_of_data"}
    for t in result.trades:
        assert t.exit_reason in valid_reasons, f"bad reason: {t.exit_reason}"
    ok(f"All {m.total_trades} trade(s) have valid exit reasons")

    report = PerformanceReport(m, title="Backtest Result", start_at=result.start_at, end_at=result.end_at)
    print(report.ascii())


# ── test 4: PerformanceAnalyzer (live DB) ────────────────────────────────────

async def test_analyzer_live():
    section("4 · PerformanceAnalyzer — live DB analysis")

    analyzer = PerformanceAnalyzer()
    report   = await analyzer.analyse_live(initial_capital=10_000.0, limit=500)

    m = report.metrics
    ok(f"DB closed trades={m.total_trades}  win_rate={m.win_rate:.1%}  pnl=${m.total_pnl:.2f}")

    if m.total_trades == 0:
        ok("No closed trades in DB yet — metrics are all-zero (expected for fresh DB)")
    else:
        assert m.gross_profit >= 0
        assert m.gross_loss >= 0

    print(report.ascii())


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    configure_logging("WARNING")
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      smart-trader · analysis layer test              ║")
    print("╚══════════════════════════════════════════════════════╝")

    test_metrics_calculator()
    test_report()
    test_backtest_engine()
    await test_analyzer_live()

    print(f"\n{'═'*56}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    asyncio.run(main())
