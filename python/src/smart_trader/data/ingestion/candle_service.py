"""CandleIngestionService — fetch OHLCV candles from Gate.io and persist them.

Three entry points:
  • backfill(symbol, timeframe, since, until)
        Pull a historical window and upsert into the DB.
  • sync(symbol, timeframe)
        Incremental: fetch from the last stored candle to now.
  • sync_all(symbols, timeframes)
        Run sync() for every symbol × timeframe combination.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from smart_trader.data.ingestion.gateio_client import TIMEFRAME_MS, GateIOClient
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory

log = structlog.get_logger(__name__)

# Default look-back window when no candles exist yet, per timeframe
DEFAULT_BACKFILL_DAYS: dict[str, int] = {
    "1m":  7,
    "5m":  14,
    "15m": 30,
    "1h":  90,
    "4h":  180,
    "1d":  365,
    "1w":  730,
}


class CandleIngestionService:
    def __init__(self, client: GateIOClient) -> None:
        self._client = client
        self._factory = get_session_factory()
        self._log = log.bind(service="candle_ingestion")

    async def backfill(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> int:
        """Fetch candles in [since, until] and persist. Returns inserted count."""
        until = until or datetime.now(timezone.utc)
        since = since or (until - timedelta(days=DEFAULT_BACKFILL_DAYS.get(timeframe, 30)))

        self._log.info("backfill_start", symbol=symbol, tf=timeframe, since=since, until=until)

        candles = await self._client.fetch_candles_paginated(
            symbol, timeframe,
            since_ms=int(since.timestamp() * 1000),
            until_ms=int(until.timestamp() * 1000),
        )

        rows = [c.to_dict() for c in candles]
        async with self._factory() as session:
            repo = CandleRepository(session)
            inserted = await repo.upsert_many(rows)

        self._log.info(
            "backfill_done",
            symbol=symbol, tf=timeframe, fetched=len(rows), inserted=inserted,
        )
        return inserted

    async def sync(self, symbol: str, timeframe: str) -> int:
        """Pull only candles newer than the last stored one.

        Falls back to a full backfill if no data exists yet.
        """
        exchange = self._client._exchange.id
        async with self._factory() as session:
            repo = CandleRepository(session)
            last_time = await repo.get_latest_time(symbol, exchange, timeframe)

        if last_time is None:
            return await self.backfill(symbol, timeframe)

        tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        since = last_time + timedelta(milliseconds=tf_ms)
        now = datetime.now(timezone.utc)

        if since >= now:
            self._log.debug("already_up_to_date", symbol=symbol, tf=timeframe)
            return 0

        return await self.backfill(symbol, timeframe, since=since, until=now)

    async def sync_all(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> dict[str, int]:
        """Sync every symbol × timeframe pair.

        Returns a mapping of ``"SYMBOL@TF"`` → inserted count (-1 on error).
        """
        results: dict[str, int] = {}
        for symbol in symbols:
            for tf in timeframes:
                key = f"{symbol}@{tf}"
                try:
                    results[key] = await self.sync(symbol, tf)
                except Exception as exc:
                    self._log.error("sync_failed", key=key, error=str(exc))
                    results[key] = -1
        return results
