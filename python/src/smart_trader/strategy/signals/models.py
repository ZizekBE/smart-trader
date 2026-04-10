"""SignalEvent — the output dataclass of the SignalEngine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode


@dataclass(slots=True)
class SignalEvent:
    symbol:        str
    timeframe:     str
    signal_type:   str            # 'long' | 'short' | 'exit'
    source:        str            # detector name, e.g. 'technical_rsi'
    raw_score:     float          # [0, 1] detector strength before context
    confidence:    float          # [0, 1] after regime / volume scoring
    regime:        MarketRegime
    strategy_mode: StrategyMode
    features:      dict[str, float]
    created_at:    datetime

    def to_db_row(self) -> dict:
        return {
            "created_at":    self.created_at,
            "symbol":        self.symbol,
            "timeframe":     self.timeframe,
            "signal_type":   self.signal_type,
            "source_model":  self.source,
            "confidence":    round(self.confidence, 4),
            "market_regime": str(self.regime),
            "strategy_mode": str(self.strategy_mode),
            "features":      {k: round(v, 6) for k, v in self.features.items()},
            "metadata":      {"raw_score": round(self.raw_score, 4)},
        }

    def __str__(self) -> str:
        return (
            f"[{self.signal_type.upper()}] {self.symbol}/{self.timeframe} "
            f"src={self.source}  conf={self.confidence:.3f}  "
            f"regime={self.regime}"
        )
