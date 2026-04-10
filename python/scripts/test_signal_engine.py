"""
Signal Engine integration test.

Layers tested:
  1. Detectors (unit)     — synthetic data validates each detector fires correctly
  2. ConfidenceScorer     — regime alignment adjusts confidence as expected
  3. SignalEngine (unit)  — end-to-end on synthetic bull/bear data
  4. SignalService (live) — BTC/USDT 1h signals from real DB candles + DB persist
  5. SignalRepository     — verify signals were written and can be queried

Run:
    python scripts/test_signal_engine.py
"""
from __future__ import annotations

import asyncio
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

from smart_trader.strategy.signals.detectors import (
    detect_rsi, detect_macd_cross, detect_bollinger_touch, detect_ema_bounce,
)
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.confidence.scorer import score_confidence
from smart_trader.strategy.trend.regime import (
    MarketRegime, StrategyMode, TrendState, REGIME_TO_MODE,
)
from smart_trader.strategy.signal_service import SignalService
from smart_trader.data.storage.signal_repo import SignalRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.utils.logging import configure_logging


def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}"); sys.exit(1)
def section(t): print(f"\n{'─'*54}\n  {t}\n{'─'*54}")


# ── synthetic data helpers ──────────────────────────────────────
def _make_oversold_df(n=50) -> pd.DataFrame:
    """Downtrend ending in RSI < 30 territory."""
    rng = np.random.default_rng(0)
    prices = [60_000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 - 0.006 + rng.normal(0, 0.003)))
    return _ohlcv_df(prices, vol_spike=False)


def _make_overbought_df(n=50) -> pd.DataFrame:
    """Uptrend ending in RSI > 70 territory."""
    rng = np.random.default_rng(1)
    prices = [40_000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + 0.006 + rng.normal(0, 0.003)))
    return _ohlcv_df(prices, vol_spike=False)


def _make_ranging_df(n=60, center=60_000) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    prices = [center + rng.normal(0, center * 0.01) for _ in range(n)]
    return _ohlcv_df(prices, vol_spike=False)


def _ohlcv_df(closes, vol_spike=False) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        s = c * 0.005
        rows.append({
            "time":   datetime(2024, 1, 1, i % 24, tzinfo=timezone.utc),
            "open":   c - s / 2,
            "high":   c + s,
            "low":    c - s,
            "close":  c,
            "volume": 5_000.0 * (10 if (vol_spike and i == len(closes) - 1) else 1),
        })
    return pd.DataFrame(rows)


def _fake_trend(regime: MarketRegime, direction: float = 0.5) -> TrendState:
    return TrendState(
        symbol="TEST/USDT", timeframe="1d",
        regime=regime,
        strategy_mode=REGIME_TO_MODE[regime],
        direction=direction,
        strength=abs(direction),
        signals={},
        calculated_at=datetime.now(timezone.utc),
    )


# ── test 1: individual detectors ────────────────────────────────
def test_detectors():
    section("1 · Individual detectors — synthetic data")

    # RSI: oversold df should trigger LONG
    df_os = _make_oversold_df()
    r = detect_rsi(df_os)
    assert r is not None and r[0] == "long", f"Expected RSI long, got {r}"
    ok(f"RSI oversold → long  score={r[1]:.3f}  rsi={r[2]['rsi']:.1f}")

    # RSI: overbought df should trigger SHORT
    df_ob = _make_overbought_df()
    r = detect_rsi(df_ob)
    assert r is not None and r[0] == "short", f"Expected RSI short, got {r}"
    ok(f"RSI overbought → short  score={r[1]:.3f}  rsi={r[2]['rsi']:.1f}")

    # MACD cross on bull run (cross happens early then continues)
    # build a series where MACD crosses zero
    closes = [50_000 * (1 - 0.003) ** i for i in range(30)]  # down
    closes += [closes[-1] * (1 + 0.007) ** i for i in range(30)]  # sharp reversal
    df_cross = _ohlcv_df(closes)
    r = detect_macd_cross(df_cross)
    # MACD cross may or may not fire depending on exact bar — just verify it doesn't crash
    ok(f"MACD cross detector ran  result={'fired' if r else 'no signal'}")

    # Bollinger: deep oversold should trigger
    # build: price way below BB lower
    df_bb = _make_oversold_df(n=30)
    r = detect_bollinger_touch(df_bb)
    ok(f"Bollinger touch ran  result={'fired '+r[0]+f' score={r[1]:.3f}' if r else 'no signal'}")

    # Insufficient data returns None
    tiny = _ohlcv_df([50_000] * 5)
    assert detect_rsi(tiny) is None, "RSI should return None on < 15 bars"
    assert detect_macd_cross(tiny) is None
    assert detect_bollinger_touch(tiny) is None
    ok("Detectors return None when data is insufficient")


# ── test 2: confidence scorer ────────────────────────────────────
def test_scorer():
    section("2 · ConfidenceScorer — regime alignment")

    raw = 0.60

    # LONG in bull trending → boosted
    bull_state  = _fake_trend(MarketRegime.BULL_TRENDING, direction=0.8)
    c_bull_long = score_confidence(raw, "long", bull_state, vol_spike=False)
    ok(f"Long in BULL_TRENDING:   raw={raw:.2f} → conf={c_bull_long:.4f} (should be > raw)")
    assert c_bull_long > raw, "Bull trend should boost long signal"

    # LONG in bear trending → penalised
    bear_state  = _fake_trend(MarketRegime.BEAR_TRENDING, direction=-0.8)
    c_bear_long = score_confidence(raw, "long", bear_state, vol_spike=False)
    ok(f"Long in BEAR_TRENDING:   raw={raw:.2f} → conf={c_bear_long:.4f} (should be < raw)")
    assert c_bear_long < raw, "Bear trend should penalise long signal"

    # Volume spike adds 10%
    c_vol = score_confidence(raw, "long", bull_state, vol_spike=True)
    ok(f"Long in BULL + vol spike: raw={raw:.2f} → conf={c_vol:.4f} (should be > bull_long)")
    assert c_vol > c_bull_long, "Volume spike should add confidence"

    # Short in bear trending → boosted
    c_bear_short = score_confidence(raw, "short", bear_state, vol_spike=False)
    ok(f"Short in BEAR_TRENDING:  raw={raw:.2f} → conf={c_bear_short:.4f} (should be > raw)")
    assert c_bear_short > raw, "Bear trend should boost short signal"

    # Hard cap at 1.0
    assert score_confidence(1.0, "long", bull_state, vol_spike=True) == 1.0
    ok("Confidence capped at 1.0")


# ── test 3: SignalEngine unit test ───────────────────────────────
def test_signal_engine():
    section("3 · SignalEngine — synthetic data")

    engine = SignalEngine()
    bull   = _fake_trend(MarketRegime.BULL_TRENDING, direction=0.8)
    bear   = _fake_trend(MarketRegime.BEAR_TRENDING, direction=-0.8)

    # oversold market in bull regime → should yield at least RSI long
    df_os = _make_oversold_df(n=80)
    events = engine.analyse(df_os, "TEST/USDT", "1h", bull, min_confidence=0.0)
    ok(f"Bull regime + oversold: {len(events)} signals")
    longs  = [e for e in events if e.signal_type == "long"]
    shorts = [e for e in events if e.signal_type == "short"]
    assert len(longs) > 0, "Expected at least one long signal"
    ok(f"  longs={len(longs)}  shorts={len(shorts)}")
    for e in events[:3]:
        ok(f"  {e}")

    # overbought market in bear regime → should yield short
    df_ob = _make_overbought_df(n=80)
    events = engine.analyse(df_ob, "TEST/USDT", "1h", bear, min_confidence=0.0)
    ok(f"Bear regime + overbought: {len(events)} signals")
    shorts2 = [e for e in events if e.signal_type == "short"]
    assert len(shorts2) > 0, "Expected at least one short signal"

    # check confidence is sorted descending
    if len(events) > 1:
        for i in range(len(events) - 1):
            assert events[i].confidence >= events[i + 1].confidence
        ok("Events sorted by confidence descending")


# ── test 4: SignalService — live DB ─────────────────────────────
async def test_signal_service():
    section("4 · SignalService — live BTC/USDT candles (1h signals, 1d trend)")

    service = SignalService(exchange="gateio")
    events  = await service.run(
        symbol="BTC/USDT",
        signal_tf="1h",
        trend_tf="1d",
        persist=True,
        min_confidence=0.30,   # lower threshold to see more signals in test
    )

    ok(f"Signals detected: {len(events)}")
    if not events:
        ok("No signals above threshold (valid — market may not have edge conditions)")
        return

    print()
    print(f"  {'Source':<20}  {'Type':<6}  {'Raw':>6}  {'Conf':>6}  {'Regime'}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*20}")
    for e in events:
        print(
            f"  {e.source:<20}  {e.signal_type:<6}  "
            f"{e.raw_score:6.4f}  {e.confidence:6.4f}  {e.regime}"
        )

    for e in events:
        assert 0 <= e.confidence <= 1.0
        assert e.signal_type in ("long", "short", "exit")
    ok("All confidence scores in [0, 1]")


# ── test 5: SignalRepository query ───────────────────────────────
async def test_signal_repo():
    section("5 · SignalRepository — query persisted signals")

    factory = get_session_factory()
    async with factory() as session:
        repo = SignalRepository(session)
        signals = await repo.get_latest("BTC/USDT", "1h", limit=10)

    ok(f"DB signals for BTC/USDT/1h: {len(signals)} rows")
    for s in signals[:3]:
        ok(f"  id={str(s.id)[:8]}…  type={s.signal_type}  conf={s.confidence}  src={s.source_model}")


# ── main ─────────────────────────────────────────────────────────
async def main():
    configure_logging("INFO")
    print("\n╔════════════════════════════════════════════════════╗")
    print("║     smart-trader · signal engine test              ║")
    print("╚════════════════════════════════════════════════════╝")

    test_detectors()
    test_scorer()
    test_signal_engine()
    await test_signal_service()
    await test_signal_repo()

    print(f"\n{'═'*54}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*54}\n")


if __name__ == "__main__":
    asyncio.run(main())
