"""Gate.io exchange client built on CCXT async.

Design mirrors the Go implementation (internal/adapters/gateio_adapter.go):
  - Retry on transient network errors (up to 3 attempts, exponential back-off)
  - Paper-mode order simulation — no real orders placed when paper=True
  - Rate limiting delegated to CCXT (enableRateLimit=True)
  - 8-decimal precision for order amounts/prices
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import structlog

log = structlog.get_logger(__name__)

EXCHANGE_ID = "gateio"

# Milliseconds per timeframe — used for pagination cursor arithmetic
TIMEFRAME_MS: dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
}

MAX_CANDLES_PER_REQUEST = 1_000
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.0


@dataclass(slots=True)
class CandleData:
    time: datetime
    symbol: str
    exchange: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "time":      self.time,
            "symbol":    self.symbol,
            "exchange":  self.exchange,
            "timeframe": self.timeframe,
            "open":      self.open,
            "high":      self.high,
            "low":       self.low,
            "close":     self.close,
            "volume":    self.volume,
        }


class GateIOClient:
    """Async Gate.io client. Use as an async context manager:

        async with GateIOClient(api_key, api_secret, paper=True) as client:
            candles = await client.fetch_candles("BTC/USDT", "1h")
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        paper: bool = True,
    ) -> None:
        self._paper = paper
        self._exchange: ccxt.gateio = ccxt.gateio(
            {
                "apiKey":          api_key,
                "secret":          api_secret,
                "enableRateLimit": True,
                "options":         {"defaultType": "spot"},
            }
        )
        self._log = log.bind(client=EXCHANGE_ID, paper=paper)

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._exchange.close()

    async def __aenter__(self) -> "GateIOClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── market data ────────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        limit: int = MAX_CANDLES_PER_REQUEST,
    ) -> list[CandleData]:
        """Fetch a single page of OHLCV bars (up to `limit`)."""

        async def _call() -> list:
            return await self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit
            )

        raw = await self._with_retry(_call, symbol=symbol, timeframe=timeframe)
        candles = [
            CandleData(
                time=datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc),
                symbol=symbol,
                exchange=EXCHANGE_ID,
                timeframe=timeframe,
                open=float(bar[1]),
                high=float(bar[2]),
                low=float(bar[3]),
                close=float(bar[4]),
                volume=float(bar[5]),
            )
            for bar in raw
        ]
        self._log.debug("candles_fetched", symbol=symbol, tf=timeframe, count=len(candles))
        return candles

    async def fetch_candles_paginated(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> list[CandleData]:
        """Paginate over the full [since_ms, until_ms] window.

        Gate.io returns at most MAX_CANDLES_PER_REQUEST bars per call; this
        method advances the cursor automatically until the window is covered.
        """
        tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
        cursor = since_ms
        all_candles: list[CandleData] = []

        while cursor < until_ms:
            batch = await self.fetch_candles(symbol, timeframe, since_ms=cursor)
            if not batch:
                break

            all_candles.extend(batch)
            last_ts_ms = int(batch[-1].time.timestamp() * 1000)

            # Stop when last returned candle reaches or passes the window end
            if last_ts_ms >= until_ms - tf_ms:
                break

            next_cursor = last_ts_ms + tf_ms
            if next_cursor <= cursor:
                break  # guard against infinite loop

            cursor = next_cursor
            await asyncio.sleep(0.1)  # gentle inter-page buffer

        until_dt = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)
        return [c for c in all_candles if c.time <= until_dt]

    async def fetch_ticker(self, symbol: str) -> dict:
        async def _call() -> dict:
            return await self._exchange.fetch_ticker(symbol)

        return await self._with_retry(_call, symbol=symbol)

    # ── order management ───────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict:
        """Place a spot order.  In paper mode, simulates the order locally."""
        if self._paper:
            fake_id = f"paper_{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}"
            self._log.info(
                "paper_order_placed",
                symbol=symbol, side=side,
                amount=f"{amount:.8f}",
                price=f"{price:.8f}" if price else "market",
                id=fake_id,
            )
            return {"id": fake_id, "status": "closed", "filled": amount, "price": price or 0.0}

        order_type = "limit" if price else "market"
        return await self._exchange.create_order(symbol, order_type, side, amount, price)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        if self._paper:
            self._log.info("paper_order_cancelled", id=order_id, symbol=symbol)
            return {"id": order_id, "status": "canceled"}
        return await self._exchange.cancel_order(order_id, symbol)

    async def get_order(self, order_id: str, symbol: str) -> dict:
        return await self._exchange.fetch_order(order_id, symbol)

    async def get_orders(self, symbol: str, status: Optional[str] = None) -> list[dict]:
        params = {"status": status} if status else {}
        return await self._exchange.fetch_orders(symbol, params=params)

    async def get_balance(self) -> dict:
        async def _call() -> dict:
            return await self._exchange.fetch_balance()

        return await self._with_retry(_call)

    # ── retry helper ───────────────────────────────────────────────────────

    async def _with_retry(self, coro_fn, **log_ctx) -> object:
        """Invoke `coro_fn()` up to MAX_RETRY_ATTEMPTS times on NetworkError."""
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                return await coro_fn()
            except ccxt.NetworkError as exc:
                if attempt == MAX_RETRY_ATTEMPTS:
                    self._log.error(
                        "network_error_final", attempt=attempt, error=str(exc), **log_ctx
                    )
                    raise
                delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                self._log.warning(
                    "network_error_retrying",
                    attempt=attempt, delay=delay, error=str(exc), **log_ctx,
                )
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as exc:
                self._log.error("exchange_error", attempt=attempt, error=str(exc), **log_ctx)
                raise
        raise RuntimeError("unreachable")  # pragma: no cover
