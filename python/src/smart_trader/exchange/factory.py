"""Exchange adapter factory — create adapters by name from Settings."""
from __future__ import annotations

from smart_trader.core.settings import get_settings
from smart_trader.exchange.adapters.ccxt_adapter import CCXTAdapter
from smart_trader.exchange.base import ExchangeAdapter
from smart_trader.exchange.models import MarketType


def create_adapter(
    exchange: str | None = None,
    market_type: MarketType = MarketType.SPOT,
    paper: bool | None = None,
) -> ExchangeAdapter:
    """Construct an ExchangeAdapter from settings.

    Args:
        exchange:    Exchange id (gateio, binance, okx, bybit).
                     Defaults to settings.default_exchange.
        market_type: spot | linear | inverse.
        paper:       Override trading_mode from settings.
    """
    s = get_settings()
    exchange = exchange or s.default_exchange
    if paper is None:
        paper = s.trading_mode != "live"

    creds = _get_credentials(exchange, s)
    return CCXTAdapter(
        exchange_id=exchange,
        api_key=creds["key"],
        api_secret=creds["secret"],
        passphrase=creds.get("passphrase", ""),
        market_type=market_type,
        paper=paper,
    )


def _get_credentials(exchange: str, s) -> dict:
    registry = {
        "gateio": {
            "key": s.gateio_api_key.get_secret_value(),
            "secret": s.gateio_api_secret.get_secret_value(),
        },
        "binance": {
            "key": s.binance_api_key.get_secret_value(),
            "secret": s.binance_api_secret.get_secret_value(),
        },
        "okx": {
            "key": s.okx_api_key.get_secret_value(),
            "secret": s.okx_api_secret.get_secret_value(),
            "passphrase": s.okx_passphrase.get_secret_value(),
        },
    }
    return registry.get(exchange, {"key": "", "secret": ""})
