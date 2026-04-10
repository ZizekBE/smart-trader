"""
Trend Engine integration test.

Layers tested:
  1. TrendEngine (unit)  — synthetic data with known regime
  2. TrendService        — reads live candles from DB, runs engine
  3. Multi-timeframe     — 1h + 4h + 1d analysis for BTC/USDT

Run:
    python scripts/test_trend_engine.py
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_USER",     "trader")
os.environ.setdefault("DB_PASSWORD", "changeme")
os.environ.setdefault("DB_NAME",     "smart_trader")

import numpy as np
import pandas as pd

from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode
from smart_trader.strategy.trend.service import TrendService
from smart_trader.utils.logging import configure_logging


def ok(msg: str):   print(f"  ✓  {msg}")
def fail(msg: str): print(f"  ✗  {msg}"); sys.exit(1)
def section(t: str): print(f"\n{'─'*52}\n  {t}\n{'─'*52}")


# ── helpers to build synthetic OHLCV DataFrames ───────────────
def _make_df(closes: list[float], vol_trend: float = 1.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price series."""
    n = len(closes)
    closes_s = pd.Series(closes, dtype=float)
    noise = 0.005  # 0.5% bar range noise
    rows = []
    base_vol = 1000.0
    for i, c in enumerate(closes):
        spread = c * noise
        rows.append({
            "time":   datetime(2024, 1, 1, tzinfo=timezone.utc),
            "open":   c - spread / 2,
            "high":   c + spread,
            "low":    c - spread,
            "close":  c,
            "volume": base_vol * (1 + vol_trend * i / n),
        })
    return pd.DataFrame(rows)


def _bull_prices(n: int = 200, start: float = 50_000) -> list[float]:
    """Steady uptrend with small noise."""
    rng = np.random.default_rng(42)
    drift = 0.003   # 0.3% per bar
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.004)))
    return prices


def _bear_prices(n: int = 200, start: float = 70_000) -> list[float]:
    """Steady downtrend with small noise."""
    rng = np.random.default_rng(42)
    drift = -0.003
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.004)))
    return prices


def _ranging_prices(n: int = 200, center: float = 60_000) -> list[float]:
    """Sideways oscillation around a fixed center."""
    rng = np.random.default_rng(42)
    prices = [center]
    for _ in range(n - 1):
        prices.append(center + rng.normal(0, center * 0.015))
    return prices


# ── test 1: unit tests on synthetic data ──────────────────────
def test_engine_unit():
    section("1 · TrendEngine — synthetic regime detection")
    engine = TrendEngine()

    # 1a: bull trend should produce direction > 0 and BULL_* regime
    df_bull = _make_df(_bull_prices(), vol_trend=0.5)
    state = engine.analyse(df_bull, "TEST/USDT", "1h")
    ok(f"Bull trend  → regime={state.regime}  dir={state.direction:+.3f}  str={state.strength:.3f}")
    assert state.direction > 0,                         "bull direction should be positive"
    assert state.regime in (MarketRegime.BULL_TRENDING, MarketRegime.BULL_RANGING), \
        f"expected bull regime, got {state.regime}"
    assert state.strategy_mode in (StrategyMode.AGGRESSIVE, StrategyMode.BALANCED), \
        f"unexpected strategy_mode {state.strategy_mode}"

    # 1b: bear trend
    df_bear = _make_df(_bear_prices(), vol_trend=0.5)
    state = engine.analyse(df_bear, "TEST/USDT", "1h")
    ok(f"Bear trend  → regime={state.regime}  dir={state.direction:+.3f}  str={state.strength:.3f}")
    assert state.direction < 0, "bear direction should be negative"
    assert state.regime in (MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_RANGING), \
        f"expected bear regime, got {state.regime}"
    assert state.strategy_mode == StrategyMode.CONSERVATIVE

    # 1c: ranging market
    df_range = _make_df(_ranging_prices(), vol_trend=0.0)
    state = engine.analyse(df_range, "TEST/USDT", "1h")
    ok(f"Ranging     → regime={state.regime}  dir={state.direction:+.3f}  str={state.strength:.3f}")
    assert abs(state.direction) < 0.6, f"ranging direction too strong: {state.direction}"

    # 1d: signal completeness
    assert set(state.signals.keys()) == {"ema_cross", "ema_long", "adx", "obv", "structure"}
    for name, score in state.signals.items():
        assert -1.0 <= score <= 1.0, f"signal {name}={score} out of [-1,1]"
    ok("All 5 signals present and within [-1, 1]")

    # 1e: insufficient data raises
    try:
        engine.analyse(df_bull.head(10), "TEST/USDT", "1h")
        fail("Should have raised ValueError for < MIN_CANDLES")
    except ValueError:
        ok("Raises ValueError when candles < MIN_CANDLES")


# ── test 2: TrendService reads from DB ────────────────────────
async def test_trend_service():
    section("2 · TrendService — live DB candles (BTC/USDT 1h)")

    service = TrendService(exchange="gateio")
    state = await service.analyse("BTC/USDT", "1h")

    ok(f"regime        = {state.regime}")
    ok(f"strategy_mode = {state.strategy_mode}")
    ok(f"direction     = {state.direction:+.4f}")
    ok(f"strength      = {state.strength:.4f}")
    ok(f"calculated_at = {state.calculated_at}")
    print()
    for name, score in state.signals.items():
        bar = "█" * int(abs(score) * 10)
        sign = "+" if score >= 0 else "-"
        print(f"    {name:<12} {sign}{bar:<10}  {score:+.4f}")

    assert state.regime is not None
    assert 0.0 <= state.strength <= 1.0
    assert -1.0 <= state.direction <= 1.0
    assert not math.isnan(state.direction)


# ── test 3: multi-timeframe analysis ─────────────────────────
async def test_multi_timeframe():
    section("3 · Multi-timeframe — BTC/USDT (1h / 4h / 1d)")

    # backfill 4h and 1d if not already in DB
    from smart_trader.data.ingestion.gateio_client import GateIOClient
    from smart_trader.data.ingestion.candle_service import CandleIngestionService

    async with GateIOClient(paper=True) as client:
        svc = CandleIngestionService(client)
        for tf in ("4h", "1d"):
            n = await svc.sync("BTC/USDT", tf)
            ok(f"sync BTC/USDT/{tf}: {n} new candles")

    service = TrendService(exchange="gateio")
    results = await service.analyse_multi("BTC/USDT", ["1h", "4h", "1d"])

    print()
    print(f"  {'TF':<6}  {'Regime':<18}  {'Mode':<13}  {'Dir':>7}  {'Str':>6}")
    print(f"  {'─'*6}  {'─'*18}  {'─'*13}  {'─'*7}  {'─'*6}")
    for tf, state in results.items():
        print(
            f"  {tf:<6}  {state.regime:<18}  {state.strategy_mode:<13}"
            f"  {state.direction:+7.4f}  {state.strength:6.4f}"
        )

    assert len(results) > 0, "no results returned"
    ok("Multi-timeframe analysis complete")


# ── main ───────────────────────────────────────────────────────
async def main():
    configure_logging("INFO")
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      smart-trader · trend engine test            ║")
    print("╚══════════════════════════════════════════════════╝")

    test_engine_unit()
    await test_trend_service()
    await test_multi_timeframe()

    print(f"\n{'═'*52}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    asyncio.run(main())
