"""CCXT-based exchange adapter — works for any CCXT-supported exchange.

Concrete subclasses only need to set `EXCHANGE_ID` and optionally override
exchange-specific quirks.  Spot and futures are handled by `market_type`.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import structlog

from smart_trader.exchange.base import ExchangeAdapter
from smart_trader.exchange.models import (
    CandleData,
    ExchangeInfo,
    FundingRate,
    MarketType,
    OpenInterest,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderResult,
    Ticker,
)

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_S = 1.0

_CCXT_TYPE_MAP = {
    MarketType.SPOT: "spot",
    MarketType.LINEAR: "swap",
    MarketType.INVERSE: "swap",
}


class CCXTAdapter(ExchangeAdapter):
    """Generic CCXT adapter.  Instantiate with exchange id string."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        market_type: MarketType = MarketType.SPOT,
        paper: bool = True,
    ) -> None:
        self._id = exchange_id
        self._market_type = market_type
        self._paper = paper

        ccxt_class = getattr(ccxt, exchange_id, None)
        if ccxt_class is None:
            raise ValueError(f"Unknown CCXT exchange: {exchange_id}")

        config: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": _CCXT_TYPE_MAP.get(market_type, "spot")},
        }
        if passphrase:
            config["password"] = passphrase

        self._ccxt: ccxt.Exchange = ccxt_class(config)
        self._log = log.bind(exchange=exchange_id, market=market_type, paper=paper)

    # ── properties ─────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return self._id

    @property
    def market_type(self) -> MarketType:
        return self._market_type

    # ── lifecycle ──────────────────────────────────────────────

    async def close(self) -> None:
        await self._ccxt.close()

    # ── REST market data ───────────────────────────────────────

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raw = await self._retry(self._ccxt.fetch_ticker, symbol)
        return Ticker(
            symbol=symbol,
            last=float(raw.get("last", 0)),
            bid=float(raw.get("bid", 0)),
            ask=float(raw.get("ask", 0)),
            volume_24h=float(raw.get("quoteVolume", 0) or raw.get("baseVolume", 0)),
            timestamp=datetime.now(timezone.utc),
        )

    async def fetch_candles(
        self, symbol: str, timeframe: str,
        since_ms: Optional[int] = None, limit: int = 1000,
    ) -> list[CandleData]:
        raw = await self._retry(
            self._ccxt.fetch_ohlcv, symbol, timeframe, since=since_ms, limit=limit,
        )
        return [
            CandleData(
                time=datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc),
                symbol=symbol, exchange=self._id, timeframe=timeframe,
                open=float(bar[1]), high=float(bar[2]),
                low=float(bar[3]), close=float(bar[4]), volume=float(bar[5]),
            )
            for bar in raw
        ]

    async def fetch_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        raw = await self._retry(self._ccxt.fetch_order_book, symbol, depth)
        return OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(p, q) for p, q in raw.get("bids", [])],
            asks=[OrderBookLevel(p, q) for p, q in raw.get("asks", [])],
            timestamp=datetime.now(timezone.utc),
        )

    async def fetch_exchange_info(self, symbol: str) -> ExchangeInfo:
        if not self._ccxt.markets:
            await self._ccxt.load_markets()
        m = self._ccxt.market(symbol)
        return ExchangeInfo(
            symbol=symbol, exchange=self._id, market_type=self._market_type,
            base=m.get("base", ""), quote=m.get("quote", ""),
            price_precision=m.get("precision", {}).get("price", 8),
            amount_precision=m.get("precision", {}).get("amount", 8),
            min_notional=float(m.get("limits", {}).get("cost", {}).get("min", 0) or 0),
            maker_fee=float(m.get("maker", 0.001) or 0.001),
            taker_fee=float(m.get("taker", 0.001) or 0.001),
        )

    # ── futures-specific ───────────────────────────────────────

    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        if self._market_type == MarketType.SPOT:
            return None
        try:
            raw = await self._retry(self._ccxt.fetch_funding_rate, symbol)
            return FundingRate(
                symbol=symbol,
                rate=float(raw.get("fundingRate", 0)),
                next_funding_time=datetime.fromtimestamp(
                    raw.get("fundingTimestamp", 0) / 1000, tz=timezone.utc,
                ),
                timestamp=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    async def fetch_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        if self._market_type == MarketType.SPOT:
            return None
        try:
            raw = await self._retry(self._ccxt.fetch_open_interest, symbol)
            return OpenInterest(
                symbol=symbol,
                value=float(raw.get("openInterestValue", 0) or raw.get("openInterest", 0)),
                timestamp=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    # ── WebSocket streams ──────────────────────────────────────

    def _has_ws(self, method: str) -> bool:
        """Check if the exchange supports a specific WebSocket method."""
        return self._ccxt.has.get(method, False) is True

    async def watch_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        if not self._has_ws("watchTicker"):
            self._log.info("ws_ticker_fallback_to_rest", exchange=self._id)
            while True:
                yield await self.fetch_ticker(symbol)
                await asyncio.sleep(1.0)
            return

        while True:
            try:
                raw = await self._ccxt.watch_ticker(symbol)
                yield Ticker(
                    symbol=symbol,
                    last=float(raw.get("last", 0)),
                    bid=float(raw.get("bid", 0)),
                    ask=float(raw.get("ask", 0)),
                    volume_24h=float(raw.get("quoteVolume", 0) or 0),
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as exc:
                self._log.warning("ws_ticker_error", error=str(exc))
                await asyncio.sleep(1.0)

    async def watch_candles(self, symbol: str, timeframe: str) -> AsyncIterator[CandleData]:
        if not self._has_ws("watchOHLCV"):
            self._log.info("ws_candles_fallback_to_rest", exchange=self._id)
            while True:
                candles = await self.fetch_candles(symbol, timeframe, limit=1)
                for c in candles:
                    yield c
                await asyncio.sleep(5.0)
            return

        while True:
            try:
                raw = await self._ccxt.watch_ohlcv(symbol, timeframe)
                for bar in raw:
                    yield CandleData(
                        time=datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc),
                        symbol=symbol, exchange=self._id, timeframe=timeframe,
                        open=float(bar[1]), high=float(bar[2]),
                        low=float(bar[3]), close=float(bar[4]), volume=float(bar[5]),
                    )
            except Exception as exc:
                self._log.warning("ws_candles_error", error=str(exc))
                await asyncio.sleep(1.0)

    async def watch_order_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        if not self._has_ws("watchOrderBook"):
            self._log.info("ws_ob_fallback_to_rest", exchange=self._id)
            while True:
                yield await self.fetch_order_book(symbol, depth)
                await asyncio.sleep(1.0)
            return

        while True:
            try:
                raw = await self._ccxt.watch_order_book(symbol, depth)
                yield OrderBookSnapshot(
                    symbol=symbol,
                    bids=[OrderBookLevel(p, q) for p, q in raw.get("bids", [])[:depth]],
                    asks=[OrderBookLevel(p, q) for p, q in raw.get("asks", [])[:depth]],
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as exc:
                self._log.warning("ws_ob_error", error=str(exc))
                await asyncio.sleep(1.0)

    # ── orders ─────────────────────────────────────────────────

    async def place_order(
        self, symbol: str, side: str, amount: float,
        price: Optional[float] = None, order_type: str = "market",
    ) -> OrderResult:
        if self._paper:
            return self._paper_fill(symbol, side, amount, price or 0.0)

        raw = await self._retry(
            self._ccxt.create_order, symbol, order_type, side, amount, price,
        )
        fill = float(raw.get("average", 0) or raw.get("price", 0) or price or 0)
        fee_info = raw.get("fee") or {}
        return OrderResult(
            order_id=str(raw.get("id", "")),
            symbol=symbol, side=side,
            quantity=float(raw.get("filled", amount)),
            fill_price=fill,
            fee=float(fee_info.get("cost", 0) or fill * amount * 0.001),
            slippage=0.0,
            status="filled" if raw.get("status") == "closed" else str(raw.get("status", "failed")),
            timestamp=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        if self._paper:
            return {"id": order_id, "status": "canceled"}
        return await self._retry(self._ccxt.cancel_order, order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._retry(self._ccxt.fetch_order, order_id, symbol)

    async def fetch_balance(self) -> dict:
        return await self._retry(self._ccxt.fetch_balance)

    # ── helpers ─────────────────────────────────────────────────

    def _paper_fill(self, symbol: str, side: str, amount: float, price: float) -> OrderResult:
        slip = price * 0.0005
        fill = (price + slip) if side == "buy" else (price - slip)
        return OrderResult(
            order_id=f"paper_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            symbol=symbol, side=side, quantity=amount,
            fill_price=round(fill, 8),
            fee=round(fill * amount * 0.001, 8),
            slippage=round(abs(fill - price) * amount, 8),
            status="filled",
            timestamp=datetime.now(timezone.utc),
        )

    async def _retry(self, fn, *args, **kwargs):
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except ccxt.NetworkError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _RETRY_BASE_S * (2 ** (attempt - 1))
                self._log.warning("retry", attempt=attempt, delay=delay, error=str(exc))
                await asyncio.sleep(delay)
            except ccxt.ExchangeError:
                raise
        raise RuntimeError("unreachable")
