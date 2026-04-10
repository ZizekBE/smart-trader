"""WebSocket feed manager — multiplexes real-time streams into callbacks.

Runs background tasks that consume WebSocket streams from ExchangeAdapter
and dispatch updates to registered listeners.  Replaces the old polling
approach with sub-second latency.

Usage::

    feed = FeedManager(adapter)
    feed.on_ticker("BTC/USDT", my_callback)
    feed.on_candle("BTC/USDT", "1m", my_candle_callback)
    await feed.start()     # spawns background tasks
    ...
    await feed.stop()
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

import structlog

from smart_trader.exchange.base import ExchangeAdapter
from smart_trader.exchange.models import CandleData, OrderBookSnapshot, Ticker

log = structlog.get_logger(__name__)

Callback = Callable[..., Coroutine[Any, Any, None]]


class FeedManager:
    """Manages WebSocket subscriptions for a single exchange adapter."""

    def __init__(self, adapter: ExchangeAdapter) -> None:
        self._adapter = adapter
        self._ticker_cbs: dict[str, list[Callback]] = defaultdict(list)
        self._candle_cbs: dict[str, list[Callback]] = defaultdict(list)
        self._ob_cbs: dict[str, list[Callback]] = defaultdict(list)
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._log = log.bind(exchange=adapter.exchange_id)

    # ── registration ───────────────────────────────────────────

    def on_ticker(self, symbol: str, cb: Callback) -> None:
        self._ticker_cbs[symbol].append(cb)

    def on_candle(self, symbol: str, timeframe: str, cb: Callback) -> None:
        key = f"{symbol}@{timeframe}"
        self._candle_cbs[key].append(cb)

    def on_order_book(self, symbol: str, cb: Callback) -> None:
        self._ob_cbs[symbol].append(cb)

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._log.info("feed_starting",
                       tickers=list(self._ticker_cbs),
                       candles=list(self._candle_cbs))

        for symbol in self._ticker_cbs:
            self._tasks.append(
                asyncio.create_task(self._run_ticker(symbol), name=f"ticker_{symbol}")
            )
        for key in self._candle_cbs:
            symbol, tf = key.split("@")
            self._tasks.append(
                asyncio.create_task(self._run_candle(symbol, tf), name=f"candle_{key}")
            )
        for symbol in self._ob_cbs:
            self._tasks.append(
                asyncio.create_task(self._run_ob(symbol), name=f"ob_{symbol}")
            )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._log.info("feed_stopped")

    # ── stream consumers ───────────────────────────────────────

    async def _run_ticker(self, symbol: str) -> None:
        async for tick in self._adapter.watch_ticker(symbol):
            if not self._running:
                break
            for cb in self._ticker_cbs[symbol]:
                try:
                    await cb(tick)
                except Exception as exc:
                    self._log.error("ticker_cb_error", symbol=symbol, error=str(exc))

    async def _run_candle(self, symbol: str, timeframe: str) -> None:
        key = f"{symbol}@{timeframe}"
        async for candle in self._adapter.watch_candles(symbol, timeframe):
            if not self._running:
                break
            for cb in self._candle_cbs[key]:
                try:
                    await cb(candle)
                except Exception as exc:
                    self._log.error("candle_cb_error", key=key, error=str(exc))

    async def _run_ob(self, symbol: str) -> None:
        async for snap in self._adapter.watch_order_book(symbol):
            if not self._running:
                break
            for cb in self._ob_cbs[symbol]:
                try:
                    await cb(snap)
                except Exception as exc:
                    self._log.error("ob_cb_error", symbol=symbol, error=str(exc))
