"""SignalEngine — version-aware dispatcher.

Selects a versioned signal strategy at construction time and delegates
all analysis to it.  The caller (TradingLoop, BacktestEngine) only needs
to know the version string; all strategy logic lives in versions/.

Usage::
    engine = SignalEngine(version="v2")                      # explicit
    engine = SignalEngine()                                  # uses LATEST_VERSION
    engine = SignalEngine(regime_routing=True)               # Phase 2.3 routing
    events = engine.analyse(df, symbol, tf, trend_state, vol_state)

Phase 2.3 regime routing (regime_routing=True):
    BULL/BEAR_TRENDING → trend_follower preset (macd + ema_bounce)
    BULL/BEAR_RANGING  → mean_reversion preset (rsi + bollinger)
    otherwise          → base strategy (all detectors)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.signals.versions import LATEST_VERSION, get_strategy
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.trend.regime import MarketRegime, TrendState

_TRENDING_REGIMES = frozenset({MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING})
_RANGING_REGIMES  = frozenset({MarketRegime.BULL_RANGING,  MarketRegime.BEAR_RANGING})


class SignalEngine:
    """Thin dispatcher — owns the strategy instance for its lifetime."""

    def __init__(
        self,
        version: str = LATEST_VERSION,
        regime_routing: bool = False,
    ) -> None:
        self._strategy       = get_strategy(version)
        self.version         = version
        self._regime_routing = regime_routing
        if regime_routing:
            self._trend_strategy: BaseSignalStrategy = get_strategy("trend_follower")
            self._range_strategy: BaseSignalStrategy = get_strategy("mean_reversion")

    def _route(self, trend_state: TrendState) -> BaseSignalStrategy:
        if trend_state.regime in _TRENDING_REGIMES:
            return self._trend_strategy
        if trend_state.regime in _RANGING_REGIMES:
            return self._range_strategy
        return self._strategy

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trend_state: TrendState,
        vol_state: Optional[object] = None,
        min_confidence: float = 0.40,
        min_votes: int = 1,
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
            min_votes:      Minimum number of independent detectors that must
                            agree on the same direction.  Default 1 (no change);
                            set to 2 for Phase-1.2 multi-source confirmation.
        """
        strategy = self._route(trend_state) if self._regime_routing else self._strategy
        events   = strategy.analyse(
            df, symbol, timeframe, trend_state, vol_state, min_confidence
        )
        if min_votes > 1:
            events = [e for e in events
                      if int(e.features.get("vote_count", 1)) >= min_votes]
        return events
