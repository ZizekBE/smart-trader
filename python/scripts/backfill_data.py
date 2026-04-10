"""Bulk data backfill script — candles + funding rates + quality checks.

Backfills historical data for RL training:
  - Multi-symbol, multi-timeframe candle data (2 years)
  - Funding rate history (for futures symbols)
  - Post-backfill data quality validation

Usage::

    # Full 2-year backfill (all configured symbols and TFs)
    cd python && uv run python scripts/backfill_data.py

    # Custom range
    uv run python scripts/backfill_data.py --since 2025-01-01 --symbols BTC/USDT

    # Only run quality checks on existing data
    uv run python scripts/backfill_data.py --quality-only

    # Include futures data (funding rates)
    uv run python scripts/backfill_data.py --futures
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk data backfill for RL training")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
                   help="Spot symbols to backfill")
    p.add_argument("--timeframes", nargs="+", default=["1m", "5m", "1h", "4h"],
                   help="Timeframes to backfill")
    p.add_argument("--since", default="2024-04-01",
                   help="Start date (YYYY-MM-DD), default: 2 years ago")
    p.add_argument("--until", default=None,
                   help="End date (YYYY-MM-DD), default: now")
    p.add_argument("--exchange", default=None,
                   help="Exchange id (default: from settings)")
    p.add_argument("--concurrency", type=int, default=2,
                   help="Max parallel fetch jobs (default: 2)")
    p.add_argument("--futures", action="store_true",
                   help="Also backfill funding rates for futures symbols")
    p.add_argument("--futures-symbols", nargs="+",
                   default=["BTC/USDT:USDT", "ETH/USDT:USDT"],
                   help="Futures symbols for funding rate backfill")
    p.add_argument("--quality-only", action="store_true",
                   help="Skip backfill, only run quality checks")
    p.add_argument("--skip-quality", action="store_true",
                   help="Skip quality checks after backfill")
    return p.parse_args()


async def main():
    args = parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until = (
        datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.until
        else datetime.now(timezone.utc)
    )

    from smart_trader.exchange.factory import create_adapter
    from smart_trader.exchange.models import MarketType

    exchange_id = args.exchange
    adapter = create_adapter(exchange=exchange_id, paper=True)

    print(f"\n{'═'*60}")
    print(f"  smart-trader · Data Backfill")
    print(f"{'═'*60}")
    print(f"  Exchange:    {adapter.exchange_id}")
    print(f"  Symbols:     {args.symbols}")
    print(f"  Timeframes:  {args.timeframes}")
    print(f"  Range:       {since.date()} → {until.date()}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"{'═'*60}\n")

    # ── candle backfill ────────────────────────────────────────
    if not args.quality_only:
        from smart_trader.data.ingestion.bulk_backfill import BulkBackfillService

        print("▶ Phase 1: Candle backfill\n")
        service = BulkBackfillService(
            adapter, concurrency=args.concurrency,
        )
        report = await service.run(
            symbols=args.symbols,
            timeframes=args.timeframes,
            since=since,
            until=until,
        )
        print(report.summary())

    # ── futures backfill ───────────────────────────────────────
    if args.futures and not args.quality_only:
        from smart_trader.data.ingestion.futures_backfill import FuturesBackfillService

        print("\n▶ Phase 2: Futures data backfill\n")
        futures_adapter = create_adapter(
            exchange=exchange_id,
            market_type=MarketType.LINEAR,
            paper=True,
        )
        futures_service = FuturesBackfillService(futures_adapter)
        fr_result = await futures_service.backfill(
            symbols=args.futures_symbols,
            since=since, until=until,
        )
        print(f"  Funding rates:  fetched={fr_result.funding_rates_fetched}  inserted={fr_result.funding_rates_inserted}")
        print(f"  Open interest:  fetched={fr_result.oi_snapshots_fetched}  inserted={fr_result.oi_snapshots_inserted}")
        if fr_result.errors:
            for e in fr_result.errors:
                print(f"  ⚠ {e}")
        await futures_adapter.close()

    # ── quality checks ─────────────────────────────────────────
    if not args.skip_quality:
        from smart_trader.data.quality import run_quality_checks

        print("\n▶ Phase 3: Data quality checks\n")
        reports = await run_quality_checks(
            symbols=args.symbols,
            exchange=adapter.exchange_id,
            timeframes=args.timeframes,
            since=since,
            until=until,
        )
        for r in reports:
            print(r.summary())

        healthy = sum(1 for r in reports if r.is_healthy)
        total = len(reports)
        print(f"\n  Quality: {healthy}/{total} datasets healthy")

    await adapter.close()
    print(f"\n{'═'*60}")
    print(f"  Done!")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
