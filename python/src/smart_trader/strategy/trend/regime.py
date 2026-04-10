"""Market regime taxonomy and the TrendState output dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MarketRegime(StrEnum):
    BULL_TRENDING = "bull_trending"   # strong uptrend, ADX > 25
    BULL_RANGING  = "bull_ranging"    # mild upward bias, low ADX
    ACCUMULATION  = "accumulation"    # sideways + rising volume → potential breakout
    DISTRIBUTION  = "distribution"    # sideways + falling volume → potential breakdown
    BEAR_RANGING  = "bear_ranging"    # mild downward bias, low ADX
    BEAR_TRENDING = "bear_trending"   # strong downtrend, ADX > 25


class StrategyMode(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"


# Default strategy mode for each regime
REGIME_TO_MODE: dict[MarketRegime, StrategyMode] = {
    MarketRegime.BULL_TRENDING: StrategyMode.AGGRESSIVE,
    MarketRegime.BULL_RANGING:  StrategyMode.BALANCED,
    MarketRegime.ACCUMULATION:  StrategyMode.BALANCED,
    MarketRegime.DISTRIBUTION:  StrategyMode.CONSERVATIVE,
    MarketRegime.BEAR_RANGING:  StrategyMode.CONSERVATIVE,
    MarketRegime.BEAR_TRENDING: StrategyMode.CONSERVATIVE,
}


@dataclass(slots=True)
class TrendState:
    symbol:        str
    timeframe:     str
    regime:        MarketRegime
    strategy_mode: StrategyMode
    direction:     float           # [-1, +1]  bearish → bullish
    strength:      float           # [ 0,  1]  weak → strong
    signals:       dict[str, float]  # per-signal scores for auditing
    calculated_at: datetime

    def __str__(self) -> str:
        sig_str = "  ".join(f"{k}={v:+.3f}" for k, v in self.signals.items())
        return (
            f"[{self.symbol}/{self.timeframe}] "
            f"regime={self.regime}  mode={self.strategy_mode}  "
            f"dir={self.direction:+.3f}  strength={self.strength:.3f}\n"
            f"  signals: {sig_str}"
        )
