"""
Candle sync integration test.

Layers tested:
  1. GateIOClient  — fetch candles from Gate.io (public API, no auth)
  2. DB connection — asyncpg / SQLAlchemy async engine
  3. CandleRepository — upsert_many + get_latest_time
  4. CandleIngestionService — full backfill → sync cycle

Run:
    python scripts/test_candle_sync.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── project source on sys.path ─────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_USER",     "trader")
os.environ.setdefault("DB_PASSWORD", "changeme")
os.environ.setdefault("DB_NAME",     "smart_trader")
os.environ.setdefault("TRADING_MODE","paper")

# ── imports after path setup ───────────────────────────────────
from smart_trader.data.ingestion.gateio_client import GateIOClient
from smart_trader.data.ingestion.candle_service import CandleIngestionService
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.utils.logging import configure_logging

SYMBOL    = "BTC/USDT"
TIMEFRAME = "1h"

# ── helpers ────────────────────────────────────────────────────
def ok(msg: str):  print(f"  ✓  {msg}")
def fail(msg: str): print(f"  ✗  {msg}"); sys.exit(1)
def section(title: str): print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


# ── test 1: Gate.io candle fetch ───────────────────────────────
async def test_gateio_fetch():
    section("1 · Gate.io candle fetch (public API)")

    async with GateIOClient(paper=True) as client:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        candles = await client.fetch_candles(
            SYMBOL, TIMEFRAME,
            since_ms=int(since.timestamp() * 1000),
            limit=5,
        )

    if not candles:
        fail("No candles returned")
    ok(f"Fetched {len(candles)} candles")

    c = candles[0]
    ok(f"First candle: time={c.time}  O={c.open}  H={c.high}  L={c.low}  C={c.close}  V={c.volume:.2f}")

    assert c.high >= c.low,         "high < low"
    assert c.high >= c.open,        "high < open"
    assert c.high >= c.close,       "high < close"
    assert c.volume > 0,            "volume == 0"
    assert c.exchange == "gateio",  f"unexpected exchange: {c.exchange}"
    ok("OHLCV sanity checks passed")
    return candles


# ── test 2: DB connection ──────────────────────────────────────
async def test_db_connection():
    section("2 · DB connection")

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            __import__("sqlalchemy").text("SELECT version()")
        )
        version = result.scalar()
    ok(f"Connected: {version[:60]}...")


# ── test 3: CandleRepository upsert ───────────────────────────
async def test_repository(candles):
    section("3 · CandleRepository upsert_many + get_latest_time")

    rows = [c.to_dict() for c in candles[:3]]
    factory = get_session_factory()

    async with factory() as session:
        repo = CandleRepository(session)
        inserted = await repo.upsert_many(rows)
    ok(f"First insert: {inserted} rows written")

    # idempotency — re-inserting same rows should insert 0
    async with factory() as session:
        repo = CandleRepository(session)
        inserted2 = await repo.upsert_many(rows)
    ok(f"Re-insert (idempotency): {inserted2} rows written (expected 0)")
    assert inserted2 == 0, f"Expected 0 re-inserts, got {inserted2}"

    # get_latest_time
    async with factory() as session:
        repo = CandleRepository(session)
        latest = await repo.get_latest_time(SYMBOL, "gateio", TIMEFRAME)
    ok(f"Latest stored time: {latest}")
    assert latest is not None, "latest_time returned None after insert"


# ── test 4: CandleIngestionService backfill + sync ─────────────
async def test_ingestion_service():
    section("4 · CandleIngestionService backfill + sync")

    async with GateIOClient(paper=True) as client:
        service = CandleIngestionService(client)

        # small backfill: last 3 days of 1h candles (~72 bars)
        since = datetime.now(timezone.utc) - timedelta(days=3)
        inserted = await service.backfill(SYMBOL, TIMEFRAME, since=since)
    ok(f"Backfill (3d, 1h): {inserted} candles inserted")
    assert inserted >= 0, "backfill returned negative"

    # incremental sync — should add 0 or a few very recent bars
    async with GateIOClient(paper=True) as client:
        service = CandleIngestionService(client)
        synced = await service.sync(SYMBOL, TIMEFRAME)
    ok(f"Incremental sync: {synced} new candles")

    # verify row count in DB
    factory = get_session_factory()
    async with factory() as session:
        import sqlalchemy as sa
        from smart_trader.data.models.candle import Candle
        result = await session.execute(
            sa.select(sa.func.count()).select_from(Candle)
            .where(Candle.symbol == SYMBOL, Candle.timeframe == TIMEFRAME)
        )
        total = result.scalar()
    ok(f"Total rows in DB for {SYMBOL}/{TIMEFRAME}: {total}")
    assert total and total > 0, "DB has no candle rows after backfill"


# ── main ───────────────────────────────────────────────────────
async def main():
    configure_logging("DEBUG")
    print("\n╔══════════════════════════════════════════════════╗")
    print("║        smart-trader · candle sync test           ║")
    print("╚══════════════════════════════════════════════════╝")

    candles = await test_gateio_fetch()
    await test_db_connection()
    await test_repository(candles)
    await test_ingestion_service()

    print(f"\n{'═'*50}")
    print("  ALL TESTS PASSED")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
