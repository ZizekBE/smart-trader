"""
Risk Layer unit + integration test.

Layers tested:
  1. PositionSizer  — Kelly sizing, ATR stop/TP, strategy-mode caps
  2. LimitsChecker  — exposure cap, duplicate-position guard
  3. CircuitBreaker — daily-loss and drawdown trip + reset
  4. RiskManager    — full evaluation pipeline (all checks in sequence)

Run:
    python scripts/test_risk_layer.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smart_trader.risk.circuit_breaker.breaker import CircuitBreaker
from smart_trader.risk.limits.checker import LimitsChecker
from smart_trader.risk.manager import RiskManager
from smart_trader.risk.models import OpenPosition, Portfolio, PositionSize
from smart_trader.risk.position.sizer import PositionSizer, _kelly_fraction
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode, REGIME_TO_MODE
from smart_trader.utils.logging import configure_logging


def ok(msg):    print(f"  ✓  {msg}")
def fail(msg):  print(f"  ✗  {msg}"); sys.exit(1)
def section(t): print(f"\n{'─'*56}\n  {t}\n{'─'*56}")


# ── shared fixtures ─────────────────────────────────────────────
PRICE = 85_000.0   # BTC price

def _portfolio(
    total=10_000.0,
    cash=10_000.0,
    daily_pnl=0.0,
    peak=None,
    positions=None,
) -> Portfolio:
    p = Portfolio(
        total_value=total,
        cash=cash,
        open_positions=positions or {},
        peak_value=peak or total,
        daily_pnl=daily_pnl,
        daily_start_value=total - daily_pnl,
    )
    return p


def _signal(
    signal_type="long",
    confidence=0.75,
    regime=MarketRegime.BULL_TRENDING,
    mode=StrategyMode.BALANCED,
) -> SignalEvent:
    return SignalEvent(
        symbol="BTC/USDT",
        timeframe="1h",
        signal_type=signal_type,
        source="technical_rsi",
        raw_score=confidence,
        confidence=confidence,
        regime=regime,
        strategy_mode=mode,
        features={"rsi": 28.0},
        created_at=datetime.now(timezone.utc),
    )


# ── test 1: Kelly fraction ──────────────────────────────────────
def test_kelly():
    section("1 · Kelly fraction formula")

    # 50% win rate, 2:1 R:R → Kelly = (0.5*2 - 0.5)/2 = 0.25
    k = _kelly_fraction(0.5, 2.0)
    ok(f"Kelly(p=0.50, rr=2): {k:.4f}  (expected ≈ 0.25)")
    assert abs(k - 0.25) < 0.001

    # 70% win rate, 2:1 R:R → Kelly = (0.7*2 - 0.3)/2 = 0.55
    k = _kelly_fraction(0.70, 2.0)
    ok(f"Kelly(p=0.70, rr=2): {k:.4f}  (expected ≈ 0.55)")
    assert abs(k - 0.55) < 0.001

    # Negative edge clamped to 0
    k = _kelly_fraction(0.30, 2.0)
    ok(f"Kelly(p=0.30, rr=2): {k:.4f}  (expected 0 — negative edge)")
    assert k == 0.0


# ── test 2: PositionSizer ───────────────────────────────────────
def test_sizer():
    section("2 · PositionSizer — sizing and SL/TP levels")

    sizer     = PositionSizer()
    portfolio = _portfolio(total=10_000, cash=10_000)

    for mode in (StrategyMode.CONSERVATIVE, StrategyMode.BALANCED, StrategyMode.AGGRESSIVE):
        sig = _signal(confidence=0.70, mode=mode)
        pos = sizer.size(sig, portfolio, PRICE)
        ok(
            f"{mode:<13}  qty={pos.quantity:.6f}  "
            f"notional=${pos.notional:.2f}  "
            f"risk%={pos.risk_pct:.3%}  "
            f"SL={pos.stop_price:.0f}  TP={pos.take_profit_price:.0f}"
        )
        assert pos.notional <= portfolio.cash,       "notional exceeds cash"
        assert pos.stop_price < PRICE,               "long SL must be < price"
        assert pos.take_profit_price > PRICE,        "long TP must be > price"
        assert pos.risk_pct > 0,                     "risk_pct must be > 0"

    # short position: SL above price, TP below
    sig_short = _signal(signal_type="short", confidence=0.70, mode=StrategyMode.BALANCED)
    pos = sizer.size(sig_short, portfolio, PRICE)
    ok(f"Short SL={pos.stop_price:.0f} (above {PRICE:.0f})  TP={pos.take_profit_price:.0f}")
    assert pos.stop_price > PRICE,           "short SL must be > price"
    assert pos.take_profit_price < PRICE,    "short TP must be < price"

    # low confidence → tiny position
    sig_low = _signal(confidence=0.51, mode=StrategyMode.BALANCED)
    pos_low = sizer.size(sig_low, portfolio, PRICE)
    sig_hi  = _signal(confidence=0.90, mode=StrategyMode.BALANCED)
    pos_hi  = sizer.size(sig_hi, portfolio, PRICE)
    ok(f"Low conf notional=${pos_low.notional:.2f}  High conf notional=${pos_hi.notional:.2f}")
    assert pos_hi.notional >= pos_low.notional, "higher confidence → larger (or equal) position"


# ── test 3: LimitsChecker ───────────────────────────────────────
def test_limits():
    section("3 · LimitsChecker — exposure and duplicate guards")

    checker = LimitsChecker(max_position_size_pct=0.05)
    portfolio = _portfolio(total=10_000, cash=10_000)

    # Normal trade: $400 on $10k = 4% → should pass
    pos_ok = PositionSize("BTC/USDT", "long", 0.005, 400.0, 83_000, 89_000, 0.01, 1_200.0)
    ok_, reasons = checker.check(portfolio, pos_ok)
    ok(f"$400 / $10k = 4%: {'PASS' if ok_ else 'FAIL'}  {reasons}")
    assert ok_, f"Expected pass, got {reasons}"

    # Over max single-position: $600 = 6% > 5% cap
    pos_big = PositionSize("BTC/USDT", "long", 0.007, 600.0, 83_000, 89_000, 0.015, 1_200.0)
    ok_, reasons = checker.check(portfolio, pos_big)
    ok(f"$600 / $10k = 6%: {'PASS' if ok_ else 'FAIL (expected)'}  {reasons}")
    assert not ok_, "Expected rejection on oversized position"

    # Duplicate position guard
    portfolio_with_pos = _portfolio(
        total=10_000, cash=9_600,
        positions={"BTC/USDT": OpenPosition("BTC/USDT", "long", 0.005, 80_000, 85_000)}
    )
    ok_, reasons = checker.check(portfolio_with_pos, pos_ok)
    ok(f"Duplicate long BTC: {'PASS' if ok_ else 'FAIL (expected)'}  {reasons}")
    assert not ok_, "Expected rejection on duplicate position"

    # Total exposure cap: near-full portfolio
    portfolio_full = _portfolio(total=10_000, cash=1_000,
        positions={"ETH/USDT": OpenPosition("ETH/USDT", "long", 10.0, 900, 900)})
    ok_, reasons = checker.check(portfolio_full, pos_ok)
    ok(f"90%+ exposure: {'PASS' if ok_ else 'FAIL (expected)'}  {reasons}")
    assert not ok_, "Expected rejection on total exposure breach"


# ── test 4: CircuitBreaker ──────────────────────────────────────
def test_circuit_breaker():
    section("4 · CircuitBreaker — trip and reset")

    cb = CircuitBreaker(max_daily_loss_pct=0.02, max_drawdown_pct=0.10)

    # Healthy portfolio → no trip
    healthy = _portfolio(total=10_000, daily_pnl=50.0, peak=10_000)
    healthy.daily_start_value = 9_950.0
    r = cb.check(healthy)
    ok(f"Healthy: tripped={r.tripped}  daily_loss={r.daily_loss_pct:.2%}  dd={r.drawdown_pct:.2%}")
    assert not r.tripped

    # Daily loss of 3% → trip
    loss_port = _portfolio(total=9_700, daily_pnl=-300.0, peak=10_000)
    loss_port.daily_start_value = 10_000.0
    r = cb.check(loss_port)
    ok(f"3% daily loss: tripped={r.tripped}  reason={r.reason}")
    assert r.tripped and r.reason == "daily_loss"

    # Reset and check drawdown trip
    cb.reset()
    assert not cb.is_tripped, "Should be open after reset"
    ok("Reset successful")

    dd_port = _portfolio(total=8_900, peak=10_000)
    r = cb.check(dd_port)
    ok(f"11% drawdown: tripped={r.tripped}  reason={r.reason}  dd={r.drawdown_pct:.2%}")
    assert r.tripped and r.reason == "drawdown"


# ── test 5: RiskManager full pipeline ───────────────────────────
def test_risk_manager():
    section("5 · RiskManager — full evaluation pipeline")

    rm = RiskManager(
        min_confidence=0.65,
        max_position_pct=0.05,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
    )
    portfolio = _portfolio(total=10_000, cash=10_000)

    # ── APPROVED: good signal, healthy portfolio ───────────────
    sig = _signal(confidence=0.75, regime=MarketRegime.BULL_TRENDING)
    dec = rm.evaluate(sig, portfolio, PRICE)
    ok(f"Approved case: {dec}")
    assert dec.approved, f"Expected approval, got: {dec.reasons}"
    assert dec.position_size is not None
    assert dec.checks["circuit_breaker"]
    assert dec.checks["confidence"]
    assert dec.checks["limits"]
    ok(f"  → qty={dec.position_size.quantity:.6f}  notional=${dec.position_size.notional:.2f}")

    # ── REJECTED: confidence below threshold ───────────────────
    sig_low = _signal(confidence=0.50)
    dec = rm.evaluate(sig_low, portfolio, PRICE)
    ok(f"Low confidence: {dec}")
    assert not dec.approved
    assert not dec.checks["confidence"]

    # ── REJECTED: circuit breaker tripped ─────────────────────
    # Simulate a bad day first
    bad_port = _portfolio(total=9_700, cash=9_700, peak=10_000)
    bad_port.daily_start_value = 10_000.0
    dec = rm.evaluate(sig, bad_port, PRICE)
    ok(f"Tripped breaker: {dec}")
    assert not dec.approved
    assert not dec.checks["circuit_breaker"]

    # Reset and confirm trading resumes
    rm.reset_breaker()
    dec = rm.evaluate(sig, portfolio, PRICE)
    ok(f"After reset: approved={dec.approved}")
    assert dec.approved

    # ── REJECTED: duplicate position ──────────────────────────
    dup_portfolio = _portfolio(
        total=10_000, cash=9_000,
        positions={"BTC/USDT": OpenPosition("BTC/USDT", "long", 0.01, 80_000, PRICE)}
    )
    dec = rm.evaluate(sig, dup_portfolio, PRICE)
    ok(f"Duplicate position: {dec}")
    assert not dec.approved


# ── main ─────────────────────────────────────────────────────────
def main():
    configure_logging("WARNING")   # suppress INFO logs for cleaner test output
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        smart-trader · risk layer test                ║")
    print("╚══════════════════════════════════════════════════════╝")

    test_kelly()
    test_sizer()
    test_limits()
    test_circuit_breaker()
    test_risk_manager()

    print(f"\n{'═'*56}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    main()
