"""StrategyRegistry — Phase 2.3 multi-strategy composition.

Maps MarketRegimes to named signal presets. Each preset is a StrategyV2
instance restricted to the detectors that work best in that regime.

Regime → preset intent
──────────────────────
BULL/BEAR_TRENDING   → trend_follower  (macd + ema_crossover — ride momentum)
BULL/BEAR_RANGING    → mean_reversion  (rsi + bollinger     — fade extremes)
ACCUMULATION         → breakout        (ema_crossover + macd — catch the launch)
DISTRIBUTION         → mean_reversion  (rsi + bollinger     — short the ceiling)
default              → full v2         (all detectors)

Usage::
    registry = StrategyRegistry()
    strategy = registry.for_regime(trend_state.regime)
    events   = strategy.analyse(df, symbol, tf, trend_state, vol_state, min_conf)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from smart_trader.strategy.signals.detectors import (
    detect_bollinger_touch,
    detect_ema_bounce,
    detect_ema_crossover,
    detect_macd_cross,
    detect_rsi,
)
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.signals.versions.v2 import StrategyV2
from smart_trader.strategy.trend.regime import MarketRegime


@dataclass
class StrategyEntry:
    name:           str
    factory:        Callable[[], BaseSignalStrategy]
    regime_affinity: frozenset[MarketRegime]
    description:    str = ""


class StrategyRegistry:
    """Stateful registry: holds one live instance per named preset."""

    def __init__(self) -> None:
        self._entries: dict[str, StrategyEntry] = {}
        self._instances: dict[str, BaseSignalStrategy] = {}
        self._regime_map: dict[MarketRegime, str] = {}
        self._default: str | None = None
        self._register_defaults()

    # ── public API ────────────────────────────────────────────────────────────

    def register(
        self,
        name:            str,
        factory:         Callable[[], BaseSignalStrategy],
        regime_affinity: frozenset[MarketRegime],
        description:     str = "",
        default:         bool = False,
    ) -> None:
        """Register a named strategy preset."""
        self._entries[name] = StrategyEntry(
            name=name,
            factory=factory,
            regime_affinity=regime_affinity,
            description=description,
        )
        for regime in regime_affinity:
            self._regime_map[regime] = name
        if default or self._default is None:
            self._default = name

    def for_regime(self, regime: MarketRegime) -> BaseSignalStrategy:
        """Return the live strategy instance for *regime* (or the default)."""
        name = self._regime_map.get(regime, self._default)
        return self._get_or_create(name)

    def get(self, name: str) -> BaseSignalStrategy:
        """Return a named preset by name."""
        if name not in self._entries:
            raise ValueError(f"Unknown strategy preset: {name!r}. "
                             f"Available: {list(self._entries)}")
        return self._get_or_create(name)

    def names(self) -> list[str]:
        return list(self._entries)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_or_create(self, name: str | None) -> BaseSignalStrategy:
        if name is None or name not in self._entries:
            raise RuntimeError("No default strategy registered")
        if name not in self._instances:
            self._instances[name] = self._entries[name].factory()
        return self._instances[name]

    def _register_defaults(self) -> None:
        _trending = frozenset({MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING})
        _breakout = frozenset({MarketRegime.ACCUMULATION})
        _ranging  = frozenset({MarketRegime.BULL_RANGING, MarketRegime.BEAR_RANGING})
        # Design principle: augment rather than restrict.
        # Presets PRIORITIZE regime-appropriate detectors (ordering matters for
        # vote-boost calculation) but keep the full set so no signal type is lost.
        # Regime-specific detectors listed first so they score highest in voting.

        self.register(
            name="trend_follower",
            factory=lambda: StrategyV2(detectors=[
                ("ema_crossover", detect_ema_crossover),  # primary: momentum crossover
                ("macd",          detect_macd_cross),     # secondary: trend confirmation
                ("ema_bounce",    detect_ema_bounce),     # tertiary: pullback entry
                ("rsi",           detect_rsi),            # guard: avoid chasing extremes
                ("bollinger",     detect_bollinger_touch),# guard: Bollinger band exits
            ]),
            regime_affinity=_trending,
            description="Trending — all detectors, ordered by momentum priority",
        )
        self.register(
            name="breakout",
            factory=lambda: StrategyV2(detectors=[
                ("ema_crossover", detect_ema_crossover),  # primary: launch signal
                ("ema_bounce",    detect_ema_bounce),     # secondary: trend structure
                ("macd",          detect_macd_cross),     # tertiary: momentum
                ("rsi",           detect_rsi),            # guard: oversold confirmation
                ("bollinger",     detect_bollinger_touch),# guard: band squeeze detection
            ]),
            regime_affinity=_breakout,
            description="Accumulation — all detectors, ordered for breakout detection",
        )
        self.register(
            name="mean_reversion",
            factory=lambda: StrategyV2(detectors=[
                ("rsi",           detect_rsi),            # primary: overbought/oversold
                ("bollinger",     detect_bollinger_touch),# secondary: band touch/reversal
                ("ema_crossover", detect_ema_crossover),  # tertiary: cross confirmation
                ("ema_bounce",    detect_ema_bounce),     # guard: structure check
                ("macd",          detect_macd_cross),     # guard: avoid fighting trend
            ]),
            regime_affinity=_ranging,
            description="Ranging — all detectors, ordered for mean-reversion",
        )
        self.register(
            name="full_v2",
            factory=lambda: StrategyV2(),  # default ordering from v2._DETECTORS
            regime_affinity=frozenset(),   # explicit fallback only
            description="Default — standard v2 ordering (distribution + unknown)",
            default=True,
        )
