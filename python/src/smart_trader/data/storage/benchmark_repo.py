"""BenchmarkRepository — writes daily snapshots and baseline for EPIC-BENCH."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smart_trader.data.models.benchmark import BenchmarkBaseline, BenchmarkSnapshot


class BenchmarkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_snapshot(
        self,
        ts:              datetime,
        symbol:          str,
        portfolio_total: float,
        cash:            float,
        bh_price:        float,
        regime:          Optional[str],
        positions:       int,
    ) -> None:
        stmt = (
            pg_insert(BenchmarkSnapshot)
            .values(
                ts=ts,
                symbol=symbol,
                portfolio_total=portfolio_total,
                cash=cash,
                bh_price=bh_price,
                regime=regime,
                positions=positions,
            )
            .on_conflict_do_update(
                index_elements=["ts", "symbol"],
                set_={
                    "portfolio_total": portfolio_total,
                    "cash": cash,
                    "bh_price": bh_price,
                    "regime": regime,
                    "positions": positions,
                },
            )
        )
        await self._s.execute(stmt)
        await self._s.commit()

    async def ensure_baseline(
        self,
        symbol:        str,
        start_price:   float,
        start_capital: float,
        start_ts:      datetime,
    ) -> None:
        """Insert baseline only if one doesn't already exist for this symbol."""
        existing = await self._s.get(BenchmarkBaseline, symbol)
        if existing is not None:
            return
        self._s.add(BenchmarkBaseline(
            symbol=symbol,
            start_price=start_price,
            start_capital=start_capital,
            start_ts=start_ts,
        ))
        await self._s.commit()

    async def get_baseline(self, symbol: str) -> Optional[BenchmarkBaseline]:
        return await self._s.get(BenchmarkBaseline, symbol)

    async def get_snapshots(
        self,
        symbol:   str,
        since:    Optional[datetime] = None,
        until:    Optional[datetime] = None,
    ) -> list[BenchmarkSnapshot]:
        stmt = (
            select(BenchmarkSnapshot)
            .where(BenchmarkSnapshot.symbol == symbol)
            .order_by(BenchmarkSnapshot.ts.asc())
        )
        if since:
            stmt = stmt.where(BenchmarkSnapshot.ts >= since)
        if until:
            stmt = stmt.where(BenchmarkSnapshot.ts <= until)
        result = await self._s.execute(stmt)
        return list(result.scalars().all())
