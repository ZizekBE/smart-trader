"""
Exchange adapter integration test — validates the new multi-exchange layer
using Gate.io (public + authenticated endpoints).

Layers tested:
  1. CCXTAdapter       — REST: ticker, candles, order book, exchange info
  2. create_adapter()  — factory builds adapter from Settings
  3. FeedManager       — WebSocket candle stream (short burst)
  4. FeatureEngine     — compute features from fetched candles
  5. OrderBookSnapshot — spread, depth calculations

Run:
    cd python && uv run python scripts/test_exchange_adapter.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def ok(msg: str):   print(f"  \u2713  {msg}")
def fail(msg: str): print(f"  \u2717  {msg}"); sys.exit(1)
def section(title: str): print(f"\n{'─'*56}\n  {title}\n{'─'*56}")


# ── test 1: create_adapter factory ─────────────────────────────
async def test_factory():
    section("1 · create_adapter() — factory from Settings")

    from smart_trader.exchange.factory import create_adapter
    from smart_trader.exchange.models import MarketType

    adapter = create_adapter()
    ok(f"Created adapter: exchange={adapter.exchange_id}, market={adapter.market_type}")
    assert adapter.exchange_id == "gateio"
    assert adapter.market_type == MarketType.SPOT
    await adapter.close()
    ok("Adapter closed cleanly")


# ── test 2: REST — ticker ──────────────────────────────────────
async def test_ticker():
    section("2 · fetch_ticker('BTC/USDT')")

    from smart_trader.exchange.factory import create_adapter

    async with create_adapter() as adapter:
        ticker = await adapter.fetch_ticker("BTC/USDT")

    assert ticker.last > 0, f"bad last price: {ticker.last}"
    assert ticker.bid > 0,  f"bad bid: {ticker.bid}"
    assert ticker.ask > 0,  f"bad ask: {ticker.ask}"
    assert ticker.ask >= ticker.bid, f"ask < bid: {ticker.ask} < {ticker.bid}"
    ok(f"last={ticker.last:,.2f}  bid={ticker.bid:,.2f}  ask={ticker.ask:,.2f}  vol24h={ticker.volume_24h:,.0f}")


# ── test 3: REST — candles ─────────────────────────────────────
async def test_candles():
    section("3 · fetch_candles('BTC/USDT', '1h', limit=10)")

    from smart_trader.exchange.factory import create_adapter

    async with create_adapter() as adapter:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        candles = await adapter.fetch_candles(
            "BTC/USDT", "1h",
            since_ms=int(since.timestamp() * 1000),
            limit=10,
        )

    assert len(candles) > 0, "no candles returned"
    ok(f"Fetched {len(candles)} candles")

    c = candles[0]
    assert c.high >= c.low,   "high < low"
    assert c.volume > 0,      "volume == 0"
    assert c.exchange == "gateio"
    ok(f"First: {c.time}  O={c.open}  H={c.high}  L={c.low}  C={c.close}  V={c.volume:.2f}")
    return candles


# ── test 4: REST — order book ──────────────────────────────────
async def test_order_book():
    section("4 · fetch_order_book('BTC/USDT', depth=10)")

    from smart_trader.exchange.factory import create_adapter

    async with create_adapter() as adapter:
        ob = await adapter.fetch_order_book("BTC/USDT", depth=10)

    assert len(ob.bids) > 0, "no bids"
    assert len(ob.asks) > 0, "no asks"
    assert ob.mid_price > 0,  "bad mid_price"
    assert ob.spread >= 0,    "negative spread"

    bid_depth, ask_depth = ob.depth_at(0.01)
    ok(f"mid={ob.mid_price:,.2f}  spread={ob.spread_bps:.2f}bps  bids={len(ob.bids)}  asks={len(ob.asks)}")
    ok(f"depth ±1%: bid_vol={bid_depth:.4f} BTC  ask_vol={ask_depth:.4f} BTC")


# ── test 5: REST — exchange info ───────────────────────────────
async def test_exchange_info():
    section("5 · fetch_exchange_info('BTC/USDT')")

    from smart_trader.exchange.factory import create_adapter

    async with create_adapter() as adapter:
        info = await adapter.fetch_exchange_info("BTC/USDT")

    ok(f"base={info.base}  quote={info.quote}  maker_fee={info.maker_fee}  taker_fee={info.taker_fee}")
    ok(f"price_prec={info.price_precision}  amount_prec={info.amount_precision}  min_notional={info.min_notional}")


# ── test 6: feature computation from live candles ──────────────
async def test_features(candles):
    section("6 · FeatureEngine — compute features from live candles")

    import pandas as pd
    from smart_trader.data.features.engine import FeatureConfig, compute_features

    records = [
        {"time": c.time, "open": c.open, "high": c.high,
         "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles
    ]
    df = pd.DataFrame(records).set_index("time")

    features = compute_features(df, FeatureConfig(), prefix="1h_")
    ok(f"Computed {len(features.columns)} feature columns from {len(df)} bars")

    sample_cols = [c for c in features.columns if any(k in c for k in ("rsi", "macd", "atr", "ema_9"))]
    last = features.iloc[-1]
    for col in sample_cols[:6]:
        ok(f"  {col} = {last[col]:.6f}")


# ── test 7: streaming — ticker (REST fallback) + candle WS ────
async def test_streaming():
    section("7 · Streaming — watch_ticker × 3 ticks (REST fallback if WS unavailable)")

    from smart_trader.exchange.factory import create_adapter

    adapter = create_adapter()
    ticks_received = []

    try:
        async for tick in adapter.watch_ticker("BTC/USDT"):
            ticks_received.append(tick)
            ok(f"tick #{len(ticks_received)}: last={tick.last:,.2f}  bid={tick.bid:,.2f}  ask={tick.ask:,.2f}")
            if len(ticks_received) >= 3:
                break
    finally:
        await adapter.close()

    assert len(ticks_received) == 3
    ok("Ticker stream working — received 3 ticks")

    section("8 · Streaming — watch_candles('BTC/USDT', '1m') × 2")

    adapter = create_adapter()
    candles_received = []

    try:
        async for candle in adapter.watch_candles("BTC/USDT", "1m"):
            candles_received.append(candle)
            ok(f"candle #{len(candles_received)}: {candle.time}  C={candle.close}  V={candle.volume:.4f}")
            if len(candles_received) >= 2:
                break
    finally:
        await adapter.close()

    assert len(candles_received) >= 2
    ok("Candle stream working")


# ── main ───────────────────────────────────────────────────────
async def main():
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║    smart-trader · exchange adapter integration test    ║")
    print("╚════════════════════════════════════════════════════════╝")

    await test_factory()
    await test_ticker()
    candles = await test_candles()
    await test_order_book()
    await test_exchange_info()
    await test_features(candles)
    await test_streaming()

    print(f"\n{'═'*56}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    asyncio.run(main())
