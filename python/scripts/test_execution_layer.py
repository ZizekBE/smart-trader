"""
Execution Layer integration test.

Layers tested:
  1. OrderManager (paper) — fill price, fee, slippage simulation
  2. TradeRepository      — open / get / close / get_open / get_closed
  3. ExecutionEngine      — process_signal + check_exits (SL & TP)
  4. get_portfolio()      — portfolio reconstruction from DB state

Run:
    python scripts/test_execution_layer.py
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

from smart_trader.execution.engine import ExecutionEngine
from smart_trader.execution.orders.manager import OrderManager, FEE_RATE, SLIPPAGE_RATE
from smart_trader.data.storage.database import get_session_factory
from smart_trader.data.storage.trade_repo import TradeRepository
from smart_trader.risk.models import Portfolio, RiskDecision, PositionSize
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode, REGIME_TO_MODE
from smart_trader.utils.logging import configure_logging


def ok(msg):    print(f"  ✓  {msg}")
def fail(msg):  print(f"  ✗  {msg}"); sys.exit(1)
def section(t): print(f"\n{'─'*56}\n  {t}\n{'─'*56}")

PRICE = 85_000.0


# ── shared fixtures ──────────────────────────────────────────────
def _signal(signal_type="long", confidence=0.75) -> SignalEvent:
    return SignalEvent(
        symbol="BTC/USDT", timeframe="1h",
        signal_type=signal_type, source="technical_rsi",
        raw_score=confidence, confidence=confidence,
        regime=MarketRegime.BULL_TRENDING,
        strategy_mode=StrategyMode.BALANCED,
        features={"rsi": 28.0},
        created_at=datetime.now(timezone.utc),
    )


def _decision(signal_type="long", notional=500.0, price=PRICE) -> RiskDecision:
    atr   = price * 0.015
    side  = signal_type
    stop  = round(price - 2 * atr, 2) if signal_type == "long" else round(price + 2 * atr, 2)
    tp    = round(price + 4 * atr, 2) if signal_type == "long" else round(price - 4 * atr, 2)
    pos   = PositionSize(
        symbol="BTC/USDT", side=side,
        quantity=round(notional / price, 8),
        notional=notional,
        stop_price=stop, take_profit_price=tp,
        risk_pct=0.02, atr=atr,
    )
    return RiskDecision(approved=True, position_size=pos,
                        reasons=["approved"], checks={})


# ── test 1: OrderManager paper fill ─────────────────────────────
async def test_order_manager():
    section("1 · OrderManager — paper fill simulation")

    om = OrderManager(paper=True)

    # BUY fill: price should be slightly above ask (slippage upward)
    r = await om.place("BTC/USDT", "buy", 0.005, PRICE)
    expected_fill = PRICE * (1 + SLIPPAGE_RATE)
    expected_fee  = expected_fill * 0.005 * FEE_RATE
    ok(f"Buy fill: price={PRICE}  fill={r.fill_price}  fee={r.fee:.6f}  slip={r.slippage:.6f}")
    assert r.fill_price > PRICE,              "buy fill should be above price"
    assert abs(r.fill_price - expected_fill) < 0.01
    assert abs(r.fee - expected_fee) < 0.0001
    assert r.status == "filled"
    assert r.order_id.startswith("paper_")

    # SELL fill: price should be slightly below
    r = await om.place("BTC/USDT", "sell", 0.005, PRICE)
    ok(f"Sell fill: price={PRICE}  fill={r.fill_price}")
    assert r.fill_price < PRICE, "sell fill should be below price"


# ── test 2: TradeRepository ──────────────────────────────────────
async def test_trade_repo():
    section("2 · TradeRepository — open / close lifecycle")

    factory = get_session_factory()

    # open
    row = {
        "opened_at":   datetime.now(timezone.utc),
        "symbol":      "BTC/USDT",
        "exchange":    "gateio",
        "side":        "buy",
        "entry_price": PRICE,
        "quantity":    0.005,
        "fees":        0.042,
        "status":      "open",
        "metadata_col": {"stop_price": 82_450.0, "take_profit_price": 90_100.0},
    }
    async with factory() as session:
        repo     = TradeRepository(session)
        trade_id = await repo.open_trade(row)
    ok(f"Opened trade: {str(trade_id)[:8]}…")

    # get
    async with factory() as session:
        repo  = TradeRepository(session)
        trade = await repo.get(trade_id)
    ok(f"Fetched trade: symbol={trade.symbol}  status={trade.status}  qty={trade.quantity}")
    assert trade.status == "open"
    assert float(trade.entry_price) == PRICE

    # get_open
    async with factory() as session:
        repo  = TradeRepository(session)
        opens = await repo.get_open("BTC/USDT")
    ok(f"Open trades for BTC/USDT: {len(opens)}")
    assert any(t.id == trade_id for t in opens)

    # close with a profitable exit
    exit_price = PRICE * 1.03   # +3%
    pnl        = (exit_price - PRICE) * 0.005 - 0.084   # entry + exit fees
    async with factory() as session:
        repo = TradeRepository(session)
        await repo.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            closed_at=datetime.now(timezone.utc),
            pnl=pnl,
            pnl_pct=pnl / (PRICE * 0.005),
            fees=0.084,
            slippage=0.021,
            exit_reason="take_profit",
        )
    async with factory() as session:
        repo  = TradeRepository(session)
        trade = await repo.get(trade_id)
    ok(f"Closed trade: status={trade.status}  pnl={float(trade.pnl):.4f}  reason={trade.exit_reason}")
    assert trade.status == "closed"
    assert trade.exit_reason == "take_profit"
    assert float(trade.pnl) > 0


# ── test 3: ExecutionEngine.process_signal ───────────────────────
async def test_engine_open():
    section("3 · ExecutionEngine — process_signal (paper)")

    engine = ExecutionEngine(paper=True)
    sig    = _signal("long")
    dec    = _decision("long", notional=400.0, price=PRICE)

    trade_id = await engine.process_signal(sig, dec, PRICE)
    ok(f"Trade opened: id={str(trade_id)[:8]}…")
    assert trade_id is not None

    # verify in DB
    factory = get_session_factory()
    async with factory() as session:
        repo  = TradeRepository(session)
        trade = await repo.get(trade_id)
    ok(f"DB record: side={trade.side}  status={trade.status}  entry={float(trade.entry_price):.2f}")
    assert trade.status == "open"
    assert trade.side == "buy"
    # fill price includes slippage, so slightly above PRICE
    assert float(trade.entry_price) >= PRICE

    return trade_id


# ── test 4: ExecutionEngine.check_exits ─────────────────────────
async def test_engine_exits(trade_id):
    section("4 · ExecutionEngine — check_exits (SL and TP)")

    # open a fresh trade for SL test
    engine = ExecutionEngine(paper=True)
    sig    = _signal("long")
    dec    = _decision("long", notional=300.0, price=PRICE)
    sl_trade_id = await engine.process_signal(sig, dec, PRICE)

    # confirm SL price
    factory = get_session_factory()
    async with factory() as session:
        repo  = TradeRepository(session)
        trade = await repo.get(sl_trade_id)
    sl_price = trade.metadata_col["stop_price"]
    ok(f"Stop-loss set at: {sl_price}")

    # simulate price crashing through SL
    crash_price = sl_price * 0.995
    closed = await engine.check_exits({"BTC/USDT": crash_price})
    ok(f"check_exits at {crash_price:.0f}: {len(closed)} trade(s) closed")
    assert len(closed) >= 1
    sl_closed = next((c for c in closed if c.trade_id == sl_trade_id), None)
    assert sl_closed is not None
    ok(f"SL triggered: pnl={sl_closed.pnl:.4f}  reason={sl_closed.exit_reason}")
    assert sl_closed.exit_reason == "stop_loss"
    assert sl_closed.pnl < 0, "SL trade should be a loss"

    # TP test: open another trade
    tp_trade_id = await engine.process_signal(sig, _decision("long", 300.0, PRICE), PRICE)
    async with factory() as session:
        repo  = TradeRepository(session)
        trade = await repo.get(tp_trade_id)
    tp_price = trade.metadata_col["take_profit_price"]
    ok(f"Take-profit set at: {tp_price}")

    # simulate price hitting TP
    moon_price = tp_price * 1.005
    closed = await engine.check_exits({"BTC/USDT": moon_price})
    tp_closed = next((c for c in closed if c.trade_id == tp_trade_id), None)
    assert tp_closed is not None
    ok(f"TP triggered: pnl={tp_closed.pnl:.4f}  reason={tp_closed.exit_reason}")
    assert tp_closed.exit_reason == "take_profit"
    assert tp_closed.pnl > 0, "TP trade should be a gain"


# ── test 5: get_portfolio() ──────────────────────────────────────
async def test_get_portfolio():
    section("5 · get_portfolio() — reconstruct state from DB")

    engine    = ExecutionEngine(paper=True)
    portfolio = await engine.get_portfolio(initial_cash=10_000.0,
                                           current_prices={"BTC/USDT": PRICE})

    ok(f"Portfolio: total=${portfolio.total_value:.2f}  cash=${portfolio.cash:.2f}")
    ok(f"  open_positions={list(portfolio.open_positions.keys())}")
    ok(f"  daily_pnl={portfolio.daily_pnl:.4f}  drawdown={portfolio.drawdown_pct:.2%}")

    assert portfolio.total_value > 0
    assert portfolio.cash >= 0
    assert 0.0 <= portfolio.drawdown_pct <= 1.0


# ── main ─────────────────────────────────────────────────────────
async def main():
    configure_logging("WARNING")
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      smart-trader · execution layer test             ║")
    print("╚══════════════════════════════════════════════════════╝")

    await test_order_manager()
    await test_trade_repo()
    trade_id = await test_engine_open()
    await test_engine_exits(trade_id)
    await test_get_portfolio()

    print(f"\n{'═'*56}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    asyncio.run(main())
