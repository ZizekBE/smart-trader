"""SignalRepository — persist and query trading signals."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smart_trader.data.models.signal import Signal


class SignalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def insert(self, row: dict) -> None:
        """Insert a single signal row (no dedup — every signal is unique)."""
        stmt = pg_insert(Signal).values([row])
        await self._s.execute(stmt)
        await self._s.commit()

    async def insert_many(self, rows: list[dict]) -> int:
        """Bulk-insert signals. Returns count inserted."""
        if not rows:
            return 0
        stmt = pg_insert(Signal).values(rows)
        result = await self._s.execute(stmt)
        await self._s.commit()
        return result.rowcount

    async def get_latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 20,
    ) -> list[Signal]:
        stmt = (
            select(Signal)
            .where(Signal.symbol == symbol, Signal.timeframe == timeframe)
            .order_by(Signal.created_at.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    async def get_since(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
    ) -> list[Signal]:
        stmt = (
            select(Signal)
            .where(
                Signal.symbol == symbol,
                Signal.timeframe == timeframe,
                Signal.created_at >= since,
            )
            .order_by(Signal.created_at.desc())
        )
        result = await self._s.execute(stmt)
        return list(result.scalars().all())
