"""ExchangeAdapter — abstract interface for all exchange integrations.

Every concrete adapter (Gate.io, Binance, OKX, Bybit) implements this ABC.
The rest of the system only depends on this interface, never on ccxt directly.
"""
from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional

from smart_trader.exchange.models import (
    CandleData,
    ExchangeInfo,
    FundingRate,
    MarketType,
    OpenInterest,
    OrderBookSnapshot,
    OrderResult,
    Ticker,
)


class ExchangeAdapter(abc.ABC):
    """Unified async interface for CEX spot + futures."""

    @property
    @abc.abstractmethod
    def exchange_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def market_type(self) -> MarketType: ...

    # ── lifecycle ──────────────────────────────────────────────

    @abc.abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "ExchangeAdapter":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── market data (REST) ─────────────────────────────────────

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker: ...

    @abc.abstractmethod
    async def fetch_candles(
        self, symbol: str, timeframe: str,
        since_ms: Optional[int] = None, limit: int = 1000,
    ) -> list[CandleData]: ...

    @abc.abstractmethod
    async def fetch_order_book(
        self, symbol: str, depth: int = 20,
    ) -> OrderBookSnapshot: ...

    @abc.abstractmethod
    async def fetch_exchange_info(self, symbol: str) -> ExchangeInfo: ...

    # ── futures-specific (default no-ops for spot adapters) ─────

    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        return None

    async def fetch_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        return None

    # ── WebSocket streams ──────────────────────────────────────

    @abc.abstractmethod
    async def watch_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Yield real-time ticker updates. Must be used as `async for`."""
        ...  # pragma: no cover
        yield  # type: ignore[misc]

    @abc.abstractmethod
    async def watch_candles(
        self, symbol: str, timeframe: str,
    ) -> AsyncIterator[CandleData]:
        ...  # pragma: no cover
        yield  # type: ignore[misc]

    async def watch_order_book(
        self, symbol: str, depth: int = 20,
    ) -> AsyncIterator[OrderBookSnapshot]:
        """Default: poll REST fallback. Override for true WebSocket."""
        while True:
            yield await self.fetch_order_book(symbol, depth)
            await asyncio.sleep(1.0)

    # ── order management ───────────────────────────────────────

    @abc.abstractmethod
    async def place_order(
        self, symbol: str, side: str, amount: float,
        price: Optional[float] = None, order_type: str = "market",
    ) -> OrderResult: ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> dict: ...

    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> dict: ...

    @abc.abstractmethod
    async def fetch_balance(self) -> dict: ...

    # ── paginated candle helper ────────────────────────────────

    async def fetch_candles_paginated(
        self, symbol: str, timeframe: str,
        since_ms: int, until_ms: int,
    ) -> list[CandleData]:
        """Paginate over [since_ms, until_ms], auto-advancing the cursor."""
        from smart_trader.data.ingestion.gateio_client import TIMEFRAME_MS
        tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        cursor, result = since_ms, []
        while cursor < until_ms:
            batch = await self.fetch_candles(symbol, timeframe, since_ms=cursor)
            if not batch:
                break
            result.extend(batch)
            last_ts = int(batch[-1].time.timestamp() * 1000)
            if last_ts >= until_ms - tf_ms:
                break
            nxt = last_ts + tf_ms
            if nxt <= cursor:
                break
            cursor = nxt
            await asyncio.sleep(0.1)
        until_dt = datetime.fromtimestamp(until_ms / 1000)
        return [c for c in result if c.time.timestamp() * 1000 <= until_ms]
