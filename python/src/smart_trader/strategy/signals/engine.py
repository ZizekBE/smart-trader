"""SignalEngine — version-aware dispatcher.

Selects a versioned signal strategy at construction time and delegates
all analysis to it.  The caller (TradingLoop, BacktestEngine) only needs
to know the version string; all strategy logic lives in versions/.

Usage::
    engine = SignalEngine(version="v2")          # explicit
    engine = SignalEngine()                      # uses LATEST_VERSION
    events = engine.analyse(df, symbol, tf, trend_state, vol_state)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.signals.versions import LATEST_VERSION, get_strategy
from smart_trader.strategy.trend.regime import TrendState


class SignalEngine:
    """Thin dispatcher — owns the strategy instance for its lifetime."""

    def __init__(self, version: str = LATEST_VERSION) -> None:
        self._strategy = get_strategy(version)
        self.version   = version

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trend_state: TrendState,
        vol_state: Optional[object] = None,
        min_confidence: float = 0.40,
    ) -> list[SignalEvent]:
        """Return ranked SignalEvents for the latest bar.

        Args:
            df:             OHLCV DataFrame (ascending time).
            symbol:         Trading pair, e.g. "BTC/USDT".
            timeframe:      Candle interval, e.g. "1h".
            trend_state:    TrendState from TrendEngine.
            vol_state:      VolatilityState (optional; passed through to strategy).
            min_confidence: Floor after all scoring; caller may raise it for
                            HIGH/CRISIS vol regimes.
        """
        return self._strategy.analyse(
            df, symbol, timeframe, trend_state, vol_state, min_confidence
        )
