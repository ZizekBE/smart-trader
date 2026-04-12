"""Bulk backfill service — parallel multi-symbol/TF candle ingestion.

Supports:
  - Resume from last stored candle (no duplicate work)
  - Chunked fetching with progress reporting
  - Configurable concurrency (rate-limit aware)
  - Post-backfill data quality validation

Usage::

    service = BulkBackfillService(adapter)
    report = await service.run(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1m", "5m", "1h", "4h"],
        since=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from smart_trader.data.ingestion.gateio_client import TIMEFRAME_MS
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.exchange.base import ExchangeAdapter

log = structlog.get_logger(__name__)

MAX_CANDLES_PER_REQUEST = 1000

# Exchange-specific limits on how far back each timeframe can go.
# Gate.io: max 10,000 candles per timeframe.
# Binance/OKX/Bybit: essentially unlimited for most timeframes.
EXCHANGE_MAX_BARS: dict[str, int] = {
    "gateio": 10_000,
}



@dataclass
class BackfillJobResult:
    symbol: str
    timeframe: str
    fetched: int = 0
    inserted: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    resumed_from: Optional[datetime] = None


@dataclass
class BackfillReport:
    jobs: list[BackfillJobResult] = field(default_factory=list)
    total_fetched: int = 0
    total_inserted: int = 0
    total_elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"  Backfill Report",
            f"{'═'*60}",
            f"  Total fetched:  {self.total_fetched:>10,}",
            f"  Total inserted: {self.total_inserted:>10,}",
            f"  Total time:     {self.total_elapsed_s:>10.1f}s",
            f"  Jobs:           {len(self.jobs):>10}",
            f"  Errors:         {len(self.errors):>10}",
            f"{'─'*60}",
        ]
        for j in self.jobs:
            status = "✗ " + j.error[:40] if j.error else "✓"
            resumed = f" (resumed from {j.resumed_from})" if j.resumed_from else ""
            lines.append(
                f"  {j.symbol:>10} {j.timeframe:>3}  "
                f"fetched={j.fetched:>8,}  inserted={j.inserted:>8,}  "
                f"{j.elapsed_s:>6.1f}s  {status}{resumed}"
            )
        lines.append(f"{'═'*60}")
        return "\n".join(lines)


class BulkBackfillService:
    """Parallel multi-symbol/TF candle backfill with resume support."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        concurrency: int = 2,
        inter_page_delay: float = 0.15,
    ) -> None:
        self._adapter = adapter
        self._concurrency = concurrency
        self._page_delay = inter_page_delay
        self._factory = get_session_factory()
        self._log = log.bind(service="bulk_backfill")

    async def run(
        self,
        symbols: list[str],
        timeframes: list[str],
        since: datetime,
        until: Optional[datetime] = None,
    ) -> BackfillReport:
        """Run backfill for all symbol × timeframe combinations.

        Uses a semaphore to limit concurrent fetches (respects exchange rate limits).
        """
        until = until or datetime.now(timezone.utc)
        report = BackfillReport()
        t0 = time.monotonic()

        sem = asyncio.Semaphore(self._concurrency)
        tasks = []
        for symbol in symbols:
            for tf in timeframes:
                tasks.append(self._backfill_one(symbol, tf, since, until, sem))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                report.errors.append(str(r))
            elif isinstance(r, BackfillJobResult):
                report.jobs.append(r)
                report.total_fetched += r.fetched
                report.total_inserted += r.inserted
                if r.error:
                    report.errors.append(f"{r.symbol}@{r.timeframe}: {r.error}")

        report.total_elapsed_s = time.monotonic() - t0
        return report

    async def _backfill_one(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
        sem: asyncio.Semaphore,
    ) -> BackfillJobResult:
        """Backfill a single symbol/timeframe pair with resume support."""
        result = BackfillJobResult(symbol=symbol, timeframe=timeframe)
        t0 = time.monotonic()

        try:
            async with sem:
                # check for existing data (resume point)
                async with self._factory() as session:
                    repo = CandleRepository(session)
                    last = await repo.get_latest_time(
                        symbol, self._adapter.exchange_id, timeframe,
                    )

                tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)

                if last is not None:
                    earliest = await repo.get_earliest_time(
                        symbol, self._adapter.exchange_id, timeframe,
                    )
                    resume_from = last + timedelta(milliseconds=tf_ms)

                    ranges: list[tuple[datetime, datetime]] = []
                    if earliest is not None and since < earliest:
                        pre_end = earliest - timedelta(milliseconds=tf_ms)
                        ranges.append((since, pre_end))
                        self._log.info("pre_fill_gap", symbol=symbol,
                                       tf=timeframe, gap_start=since, gap_end=pre_end)
                    if resume_from < until:
                        ranges.append((resume_from, until))
                        result.resumed_from = resume_from

                    if not ranges:
                        self._log.info("already_complete", symbol=symbol, tf=timeframe)
                        result.elapsed_s = time.monotonic() - t0
                        return result
                else:
                    ranges = [(since, until)]

                # clamp to exchange max history limit (with 5% safety margin)
                max_bars = EXCHANGE_MAX_BARS.get(self._adapter.exchange_id, 0)
                clamped_ranges: list[tuple[datetime, datetime]] = []
                for r_since, r_until in ranges:
                    if max_bars > 0:
                        max_lookback_ms = int(max_bars * 0.95) * tf_ms
                        earliest_allowed = r_until - timedelta(milliseconds=max_lookback_ms)
                        if r_since < earliest_allowed:
                            self._log.warning(
                                "clamped_to_exchange_limit",
                                symbol=symbol, tf=timeframe,
                                requested=r_since.isoformat(),
                                clamped=earliest_allowed.isoformat(),
                                max_bars=max_bars,
                            )
                            r_since = earliest_allowed
                    clamped_ranges.append((r_since, r_until))

                total_fetched, total_inserted = 0, 0
                for r_since, r_until in clamped_ranges:
                    fetched, inserted = await self._fetch_and_store(
                        symbol, timeframe, r_since, r_until,
                    )
                    total_fetched += fetched
                    total_inserted += inserted
                result.fetched = total_fetched
                result.inserted = total_inserted

        except Exception as exc:
            result.error = str(exc)
            self._log.error("backfill_error", symbol=symbol, tf=timeframe, error=str(exc))

        result.elapsed_s = time.monotonic() - t0
        return result

    async def _fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> tuple[int, int]:
        """Paginate through the exchange API and store chunks incrementally."""
        tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
        cursor_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        total_fetched = 0
        total_inserted = 0
        page = 0

        expected_bars = (until_ms - cursor_ms) // tf_ms
        self._log.info(
            "backfill_starting",
            symbol=symbol, tf=timeframe,
            since=since.isoformat(), until=until.isoformat(),
            expected_bars=expected_bars,
        )

        while cursor_ms < until_ms:
            batch = await self._adapter.fetch_candles(
                symbol, timeframe, since_ms=cursor_ms, limit=MAX_CANDLES_PER_REQUEST,
            )
            if not batch:
                break

            # filter to window
            rows = [
                c.to_dict() for c in batch
                if int(c.time.timestamp() * 1000) <= until_ms
            ]

            if rows:
                async with self._factory() as session:
                    repo = CandleRepository(session)
                    inserted = await repo.upsert_many(rows)
                total_inserted += inserted

            total_fetched += len(rows)
            page += 1

            last_ts_ms = int(batch[-1].time.timestamp() * 1000)
            if last_ts_ms >= until_ms - tf_ms:
                break
            next_cursor = last_ts_ms + tf_ms
            if next_cursor <= cursor_ms:
                break
            cursor_ms = next_cursor

            if page % 10 == 0:
                pct = min(100, (cursor_ms - int(since.timestamp() * 1000)) / (until_ms - int(since.timestamp() * 1000) + 1) * 100)
                self._log.info(
                    "backfill_progress",
                    symbol=symbol, tf=timeframe,
                    page=page, fetched=total_fetched,
                    progress=f"{pct:.1f}%",
                )

            await asyncio.sleep(self._page_delay)

        self._log.info(
            "backfill_complete",
            symbol=symbol, tf=timeframe,
            fetched=total_fetched, inserted=total_inserted,
        )
        return total_fetched, total_inserted
