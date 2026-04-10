"""Strategy v1 — baseline: 4 detectors + regime/volume confidence scoring.

This is the original SignalEngine logic extracted verbatim so it can be
selected explicitly for comparison or rollback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from smart_trader.strategy.confidence.scorer import score_confidence
from smart_trader.strategy.signals.detectors import (
    DetectorResult,
    detect_bollinger_touch,
    detect_ema_bounce,
    detect_macd_cross,
    detect_rsi,
    volume_spike,
)
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.trend.regime import TrendState

_DETECTORS: list[tuple[str, callable]] = [
    ("rsi",       detect_rsi),
    ("macd",      detect_macd_cross),
    ("bollinger", detect_bollinger_touch),
    ("ema_bounce",detect_ema_bounce),
]

MIN_RAW_SCORE = 0.40


class StrategyV1(BaseSignalStrategy):
    """Baseline strategy: independent detectors, bool volume spike, regime factor."""

    VERSION = "v1"

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trend_state: TrendState,
        vol_state: Optional[object],
        min_confidence: float,
    ) -> list[SignalEvent]:
        if df.empty:
            return []

        vol_spiked = volume_spike(df)
        now        = datetime.now(timezone.utc)
        events: list[SignalEvent] = []

        for name, detector in _DETECTORS:
            result: DetectorResult = detector(df)
            if result is None:
                continue

            signal_type, raw_score, features = result
            if raw_score < MIN_RAW_SCORE:
                continue

            features["volume_spike"] = float(vol_spiked)

            confidence = score_confidence(
                raw_score=raw_score,
                signal_type=signal_type,
                trend_state=trend_state,
                vol_spike=vol_spiked,
            )
            if confidence < min_confidence:
                continue

            events.append(
                SignalEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal_type,
                    source=f"technical_{name}",
                    raw_score=round(raw_score, 4),
                    confidence=round(confidence, 4),
                    regime=trend_state.regime,
                    strategy_mode=trend_state.strategy_mode,
                    features=features,
                    created_at=now,
                )
            )

        events.sort(key=lambda e: e.confidence, reverse=True)
        return events
