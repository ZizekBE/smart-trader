"""BaseSignalStrategy — abstract contract for all versioned strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.trend.regime import TrendState


class BaseSignalStrategy(ABC):
    """Every strategy version must implement this interface.

    Versioning contract:
    - VERSION must be a unique, monotonically increasing string (e.g. "v1", "v2").
    - analyse() is the single entry point; it must be stateless.
    - vol_state is typed as Optional[object] to avoid a hard import cycle;
      callers pass the VolatilityState dataclass when available.
    """

    VERSION: str = "base"

    @abstractmethod
    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trend_state: TrendState,
        vol_state: Optional[object],
        min_confidence: float,
    ) -> list[SignalEvent]:
        """Return ranked SignalEvents for the latest bar in `df`.

        Args:
            df:             OHLCV DataFrame, ascending time index.
            symbol:         e.g. "BTC/USDT".
            timeframe:      e.g. "1h".
            trend_state:    TrendState from TrendEngine.
            vol_state:      VolatilityState (may be None if analysis failed).
            min_confidence: Caller-supplied floor (may be raised by vol regime).

        Returns:
            List sorted descending by confidence; may be empty.
        """
