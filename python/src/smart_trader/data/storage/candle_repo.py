"""CandleRepository — all DB operations for the `candles` hypertable."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smart_trader.data.models.candle import Candle


class CandleRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_many(self, rows: list[dict], chunk_size: int = 500) -> int:
        """Bulk-insert candles, skipping duplicates via uq_candle constraint.

        Inserts in chunks to respect asyncpg's 32 767 bind-parameter limit.
        Returns the number of rows actually inserted (0 if all were duplicates).
        """
        if not rows:
            return 0
        total = 0
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(Candle).values(chunk)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_candle")
            result = await self._s.execute(stmt)
            total += result.rowcount
        await self._s.commit()
        return total

    async def get_latest_time(
        self, symbol: str, exchange: str, timeframe: str
    ) -> Optional[datetime]:
        """Return the timestamp of the most-recent stored candle, or None."""
        stmt = (
            select(Candle.time)
            .where(
                Candle.symbol == symbol,
                Candle.exchange == exchange,
                Candle.timeframe == timeframe,
            )
            .order_by(Candle.time.desc())
            .limit(1)
        )
        result = await self._s.execute(stmt)
        return result.scalar_one_or_none()

    async def get_range(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> list[Candle]:
        """Return all candles in [since, until] ordered by time ascending."""
        stmt = (
            select(Candle)
            .where(
                Candle.symbol == symbol,
                Candle.exchange == exchange,
                Candle.timeframe == timeframe,
                Candle.time >= since,
                Candle.time <= until,
            )
            .order_by(Candle.time)
        )
        result = await self._s.execute(stmt)
        return list(result.scalars().all())
