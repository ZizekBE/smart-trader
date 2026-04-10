"""TradeRepository — CRUD for the `trades` and `portfolio_state` tables."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smart_trader.data.models.portfolio_state import PortfolioState
from smart_trader.data.models.trade import Trade


class TradeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def open_trade(self, row: dict) -> uuid.UUID:
        """Insert a new open trade. Returns the generated UUID."""
        trade = Trade(**row)
        self._s.add(trade)
        await self._s.commit()
        await self._s.refresh(trade)
        return trade.id

    async def get_for_update(self, trade_id: uuid.UUID) -> Optional[Trade]:
        """Fetch a trade with a row-level lock (SELECT ... FOR UPDATE).

        The lock is held until the enclosing transaction commits or rolls back,
        preventing concurrent close_trade calls from double-closing.
        """
        stmt = (
            select(Trade)
            .where(Trade.id == trade_id)
            .with_for_update()
        )
        result = await self._s.execute(stmt)
        return result.scalar_one_or_none()

    async def close_trade(
        self,
        trade_id:   uuid.UUID,
        exit_price: float,
        closed_at:  datetime,
        pnl:        float,
        pnl_pct:    float,
        fees:       float,
        slippage:   float,
        exit_reason: str,
    ) -> bool:
        """Atomically close a trade that is still open.

        Returns True if the row was updated, False if the trade was already
        closed (or does not exist) — prevents double-close races.
        """
        stmt = (
            update(Trade)
            .where(Trade.id == trade_id, Trade.status == "open")
            .values(
                exit_price=exit_price,
                closed_at=closed_at,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=fees,
                slippage=slippage,
                status="closed",
                exit_reason=exit_reason,
            )
        )
        result = await self._s.execute(stmt)
        await self._s.commit()
        return result.rowcount > 0

    async def cancel_trade(self, trade_id: uuid.UUID) -> None:
        stmt = (
            update(Trade)
            .where(Trade.id == trade_id)
            .values(status="cancelled", closed_at=datetime.utcnow())
        )
        await self._s.execute(stmt)
        await self._s.commit()

    async def get_open(
        self,
        symbol: Optional[str] = None,
        sleeve: Optional[str] = None,
    ) -> list[Trade]:
        stmt = select(Trade).where(Trade.status == "open")
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)
        if sleeve:
            stmt = stmt.where(Trade.sleeve == sleeve)
        stmt = stmt.order_by(Trade.opened_at.desc())
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    async def get(self, trade_id: uuid.UUID) -> Optional[Trade]:
        result = await self._s.execute(select(Trade).where(Trade.id == trade_id))
        return result.scalar_one_or_none()

    async def get_closed(
        self,
        symbol:  Optional[str]      = None,
        since:   Optional[datetime] = None,
        until:   Optional[datetime] = None,
        limit:   int                = 50,
    ) -> list[Trade]:
        stmt = select(Trade).where(Trade.status == "closed")
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)
        if since:
            stmt = stmt.where(Trade.closed_at >= since)
        if until:
            stmt = stmt.where(Trade.closed_at <= until)
        stmt = stmt.order_by(Trade.closed_at.desc()).limit(limit)
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    # ── portfolio peak value ──────────────────────────────────

    async def get_peak_value(self, key: str = "default") -> Optional[float]:
        """Read the persisted portfolio peak value."""
        stmt = select(PortfolioState.peak_value).where(PortfolioState.key == key)
        result = await self._s.execute(stmt)
        return result.scalar_one_or_none()

    async def update_peak_value(self, value: float, key: str = "default") -> None:
        """Set peak_value to max(existing, value) using upsert."""
        now = datetime.now(timezone.utc)
        stmt = (
            pg_insert(PortfolioState)
            .values(key=key, peak_value=value, updated_at=now)
            .on_conflict_do_update(
                index_elements=[PortfolioState.key],
                set_={"peak_value": value, "updated_at": now},
            )
        )
        await self._s.execute(stmt)
        await self._s.commit()
