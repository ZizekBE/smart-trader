"""Exchange-agnostic data models shared across all adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MarketType(StrEnum):
    SPOT = "spot"
    LINEAR = "linear"      # USDT-margined futures
    INVERSE = "inverse"    # coin-margined futures


@dataclass(slots=True)
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: datetime


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        return (self.spread / mid * 10_000) if mid > 0 else 0.0

    def depth_at(self, pct: float = 0.01) -> tuple[float, float]:
        """Cumulative bid/ask volume within `pct` of mid price."""
        mid = self.mid_price
        if mid <= 0:
            return 0.0, 0.0
        bid_vol = sum(l.quantity for l in self.bids if l.price >= mid * (1 - pct))
        ask_vol = sum(l.quantity for l in self.asks if l.price <= mid * (1 + pct))
        return bid_vol, ask_vol


@dataclass(slots=True)
class FundingRate:
    symbol: str
    rate: float               # current period rate (e.g. 0.0001 = 0.01%)
    next_funding_time: datetime
    timestamp: datetime


@dataclass(slots=True)
class OpenInterest:
    symbol: str
    value: float              # in quote currency (USDT)
    timestamp: datetime


@dataclass(slots=True)
class CandleData:
    """Unified OHLCV candle."""
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
            "time": self.time, "symbol": self.symbol,
            "exchange": self.exchange, "timeframe": self.timeframe,
            "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": self.volume,
        }


@dataclass(slots=True)
class OrderResult:
    order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fee: float
    slippage: float
    status: str          # filled | partial | failed
    timestamp: datetime


@dataclass
class ExchangeInfo:
    """Static metadata about a trading pair on an exchange."""
    symbol: str
    exchange: str
    market_type: MarketType
    base: str
    quote: str
    price_precision: int
    amount_precision: int
    min_notional: float = 0.0
    max_leverage: float = 1.0
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    contract_size: float = 1.0
