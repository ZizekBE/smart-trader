"""SignalEngine — version-aware dispatcher with optional regime routing.

Usage::
    engine = SignalEngine(version="v2")                      # explicit version
    engine = SignalEngine()                                  # uses LATEST_VERSION
    engine = SignalEngine(regime_routing=True)               # Phase 2.3 routing
    events = engine.analyse(df, symbol, tf, trend_state, vol_state)

Phase 2.3 regime routing (regime_routing=True):
    BULL/BEAR_TRENDING   → trend_follower (macd + ema_crossover)
    BULL/BEAR_RANGING    → mean_reversion (rsi + bollinger)
    ACCUMULATION         → breakout       (ema_crossover + macd + ema_bounce)
    DISTRIBUTION         → mean_reversion (rsi + bollinger)
    otherwise            → full_v2        (all detectors)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.signals.versions import LATEST_VERSION, get_strategy
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.trend.regime import TrendState


class SignalEngine:
    """Thin dispatcher — owns one strategy instance per preset lifetime."""

    def __init__(
        self,
        version:        str  = LATEST_VERSION,
        regime_routing: bool = False,
    ) -> None:
        self._strategy       = get_strategy(version)
        self.version         = version
        self._regime_routing = regime_routing
        self._registry       = None
        if regime_routing:
            from smart_trader.strategy.registry import StrategyRegistry
            self._registry = StrategyRegistry()

    def _route(self, trend_state: TrendState) -> BaseSignalStrategy:
        if self._registry is not None:
            return self._registry.for_regime(trend_state.regime)
        return self._strategy

    def analyse(
        self,
        df:             pd.DataFrame,
        symbol:         str,
        timeframe:      str,
        trend_state:    TrendState,
        vol_state:      Optional[object] = None,
        min_confidence: float = 0.40,
        min_votes:      int   = 1,
    ) -> list[SignalEvent]:
        """Return ranked SignalEvents for the latest bar."""
        strategy = self._route(trend_state) if self._regime_routing else self._strategy
        events   = strategy.analyse(
            df, symbol, timeframe, trend_state, vol_state, min_confidence
        )
        if min_votes > 1:
            events = [e for e in events
                      if int(e.features.get("vote_count", 1)) >= min_votes]
        return events
