"""Strategy version registry.

Adding a new version:
    1. Create versions/vN.py with a class that inherits BaseSignalStrategy.
    2. Set VERSION = "vN" on the class.
    3. Import and register it here.

Switching version at runtime:
    Set STRATEGY_VERSION=vN in .env (or the environment).
    The SignalEngine reads this via Settings.strategy_version.

Named presets (Phase 2.3 multi-strategy composition):
    "trend_follower" — StrategyV2 restricted to macd + ema_bounce detectors.
                       Activated in BULL/BEAR_TRENDING regimes.
    "mean_reversion" — StrategyV2 restricted to rsi + bollinger detectors.
                       Activated in BULL/BEAR_RANGING regimes.
"""
from __future__ import annotations

from typing import Callable

from smart_trader.strategy.signals.detectors import (
    detect_bollinger_touch,
    detect_ema_bounce,
    detect_macd_cross,
    detect_rsi,
)
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.signals.versions.v1 import StrategyV1
from smart_trader.strategy.signals.versions.v2 import StrategyV2

# ── registry ──────────────────────────────────────────────────────────────────
REGISTRY: dict[str, type[BaseSignalStrategy]] = {
    StrategyV1.VERSION: StrategyV1,
    StrategyV2.VERSION: StrategyV2,
}

# Named presets — keyed by string, value is a zero-arg factory.
# These are routed automatically by SignalEngine when regime_routing=True.
_PRESETS: dict[str, Callable[[], BaseSignalStrategy]] = {
    "trend_follower": lambda: StrategyV2(detectors=[
        ("macd",       detect_macd_cross),
        ("ema_bounce", detect_ema_bounce),
    ]),
    "mean_reversion": lambda: StrategyV2(detectors=[
        ("rsi",       detect_rsi),
        ("bollinger", detect_bollinger_touch),
    ]),
}

LATEST_VERSION = StrategyV2.VERSION


def get_strategy(version: str) -> BaseSignalStrategy:
    """Instantiate and return the strategy for *version* or a named preset.

    Raises:
        ValueError: if *version* is not in the registry or presets.
    """
    if version in _PRESETS:
        return _PRESETS[version]()
    if version not in REGISTRY:
        available = ", ".join(sorted({*REGISTRY, *_PRESETS}))
        raise ValueError(
            f"Unknown strategy version {version!r}. Available: {available}"
        )
    return REGISTRY[version]()


__all__ = ["BaseSignalStrategy", "REGISTRY", "LATEST_VERSION", "get_strategy"]
