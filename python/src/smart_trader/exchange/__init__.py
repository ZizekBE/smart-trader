"""Multi-exchange abstraction layer.

Provides a unified async interface for spot + futures across CEX platforms
(Gate.io, Binance, OKX, Bybit).  Callers work with exchange-agnostic models
and never import ccxt directly.
"""
from smart_trader.exchange.base import ExchangeAdapter
from smart_trader.exchange.factory import create_adapter
from smart_trader.exchange.models import (
    FundingRate,
    OrderBookSnapshot,
    Ticker,
)

__all__ = [
    "ExchangeAdapter",
    "create_adapter",
    "FundingRate",
    "OrderBookSnapshot",
    "Ticker",
]
