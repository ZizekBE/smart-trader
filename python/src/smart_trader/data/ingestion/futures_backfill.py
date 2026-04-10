"""Futures data backfill — funding rates and open interest history.

Fetches historical funding rate settlements and OI snapshots via
the ExchangeAdapter and stores them in TimescaleDB.

Usage::

    service = FuturesBackfillService(adapter)
    fr_count, oi_count = await service.backfill(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        since=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smart_trader.data.models.funding_rate import FundingRateRecord
from smart_trader.data.models.open_interest import OpenInterestRecord
from smart_trader.data.storage.database import get_session_factory
from smart_trader.exchange.base import ExchangeAdapter

log = structlog.get_logger(__name__)

FUNDING_INTERVAL_H = 8  # most CEXs settle funding every 8 hours


@dataclass
class FuturesBackfillResult:
    funding_rates_fetched: int = 0
    funding_rates_inserted: int = 0
    oi_snapshots_fetched: int = 0
    oi_snapshots_inserted: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class FuturesBackfillService:
    """Backfills funding rate and open interest history."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        page_delay: float = 0.2,
    ) -> None:
        self._adapter = adapter
        self._page_delay = page_delay
        self._factory = get_session_factory()
        self._log = log.bind(
            service="futures_backfill",
            exchange=adapter.exchange_id,
        )

    async def backfill(
        self,
        symbols: list[str],
        since: datetime,
        until: Optional[datetime] = None,
    ) -> FuturesBackfillResult:
        """Backfill funding rates and OI for all given futures symbols."""
        until = until or datetime.now(timezone.utc)
        result = FuturesBackfillResult()

        for symbol in symbols:
            try:
                fr = await self._backfill_funding_rates(symbol, since, until)
                result.funding_rates_fetched += fr[0]
                result.funding_rates_inserted += fr[1]
            except Exception as exc:
                msg = f"funding_rate {symbol}: {exc}"
                result.errors.append(msg)
                self._log.error("fr_backfill_error", symbol=symbol, error=str(exc))

            try:
                oi = await self._backfill_open_interest(symbol, since, until)
                result.oi_snapshots_fetched += oi[0]
                result.oi_snapshots_inserted += oi[1]
            except Exception as exc:
                msg = f"open_interest {symbol}: {exc}"
                result.errors.append(msg)
                self._log.error("oi_backfill_error", symbol=symbol, error=str(exc))

        self._log.info(
            "futures_backfill_complete",
            fr_fetched=result.funding_rates_fetched,
            fr_inserted=result.funding_rates_inserted,
            oi_fetched=result.oi_snapshots_fetched,
            oi_inserted=result.oi_snapshots_inserted,
            errors=len(result.errors),
        )
        return result

    async def _backfill_funding_rates(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
    ) -> tuple[int, int]:
        """Fetch funding rate history via CCXT's fetchFundingRateHistory."""
        exchange_id = self._adapter.exchange_id
        self._log.info("fr_backfill_start", symbol=symbol)

        all_rows = []
        cursor_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        # CCXT fetchFundingRateHistory returns historical settlements
        ccxt_obj = getattr(self._adapter, "_ccxt", None)
        if ccxt_obj is None or not ccxt_obj.has.get("fetchFundingRateHistory", False):
            # fallback: just fetch current rate
            fr = await self._adapter.fetch_funding_rate(symbol)
            if fr:
                all_rows.append({
                    "time": fr.timestamp,
                    "symbol": symbol,
                    "exchange": exchange_id,
                    "rate": fr.rate,
                    "next_funding": fr.next_funding_time,
                })
        else:
            while cursor_ms < until_ms:
                try:
                    raw = await ccxt_obj.fetch_funding_rate_history(
                        symbol, since=cursor_ms, limit=100,
                    )
                except Exception:
                    break

                if not raw:
                    break

                for entry in raw:
                    ts = entry.get("timestamp", 0)
                    if ts > until_ms:
                        continue
                    all_rows.append({
                        "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                        "symbol": symbol,
                        "exchange": exchange_id,
                        "rate": float(entry.get("fundingRate", 0)),
                        "next_funding": None,
                    })

                last_ts = raw[-1].get("timestamp", 0)
                if last_ts <= cursor_ms:
                    break
                cursor_ms = last_ts + 1
                await asyncio.sleep(self._page_delay)

        if not all_rows:
            return 0, 0

        inserted = await self._upsert_funding_rates(all_rows)
        return len(all_rows), inserted

    async def _backfill_open_interest(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
    ) -> tuple[int, int]:
        """Fetch OI snapshots. Most exchanges only provide current OI,
        so we store what we can and build history over time."""
        exchange_id = self._adapter.exchange_id
        self._log.info("oi_backfill_start", symbol=symbol)

        oi = await self._adapter.fetch_open_interest(symbol)
        if oi is None:
            return 0, 0

        rows = [{
            "time": oi.timestamp,
            "symbol": symbol,
            "exchange": exchange_id,
            "value_usd": oi.value,
        }]

        inserted = await self._upsert_open_interest(rows)
        return 1, inserted

    async def _upsert_funding_rates(self, rows: list[dict]) -> int:
        async with self._factory() as session:
            total = 0
            for i in range(0, len(rows), 500):
                chunk = rows[i:i + 500]
                stmt = pg_insert(FundingRateRecord).values(chunk)
                stmt = stmt.on_conflict_do_nothing(constraint="idx_fr_uniq")
                result = await session.execute(stmt)
                total += result.rowcount
            await session.commit()
        return total

    async def _upsert_open_interest(self, rows: list[dict]) -> int:
        async with self._factory() as session:
            stmt = pg_insert(OpenInterestRecord).values(rows)
            stmt = stmt.on_conflict_do_nothing(constraint="idx_oi_uniq")
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
