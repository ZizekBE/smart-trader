"""Strategy version registry.

Adding a new version:
    1. Create versions/vN.py with a class that inherits BaseSignalStrategy.
    2. Set VERSION = "vN" on the class.
    3. Import and register it here.

Switching version at runtime:
    Set STRATEGY_VERSION=vN in .env (or the environment).
    The SignalEngine reads this via Settings.strategy_version.
"""
from __future__ import annotations

from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.signals.versions.v1 import StrategyV1
from smart_trader.strategy.signals.versions.v2 import StrategyV2

# ── registry ──────────────────────────────────────────────────────────────────
REGISTRY: dict[str, type[BaseSignalStrategy]] = {
    StrategyV1.VERSION: StrategyV1,
    StrategyV2.VERSION: StrategyV2,
}

LATEST_VERSION = StrategyV2.VERSION


def get_strategy(version: str) -> BaseSignalStrategy:
    """Instantiate and return the strategy for *version*.

    Raises:
        ValueError: if *version* is not in the registry.
    """
    if version not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown strategy version {version!r}. Available: {available}"
        )
    return REGISTRY[version]()


__all__ = ["BaseSignalStrategy", "REGISTRY", "LATEST_VERSION", "get_strategy"]
