"""Strategy v2 — multi-signal voting + VWAP alignment + continuous volume scoring.

Improvements over v1:

1. VWAP alignment factor
   Rolling 24-bar VWAP replaces a static price level.  Signals entering
   near or through VWAP receive a confidence bonus; signals chasing price
   far away from VWAP are penalised.

2. Continuous volume-ratio factor
   Replaces the binary volume_spike bool.  The ratio of the last bar's
   volume to the 20-bar average is mapped to a smooth [0.85, 1.15]
   multiplier so moderate confirming volume is also rewarded.

3. Multi-signal voting
   After all detectors run, signals are grouped by direction.  When two
   or more independent detectors agree, each event's confidence is boosted
   (×1.20 for 2 votes, ×1.40 for 3+).  Single-source signals face a
   raised min_confidence floor (MIN_CONF_SINGLE = 0.65) to reduce noise.

Audit trail:
   Every SignalEvent.features dict gains:
     vote_count    — number of detectors that agreed on this direction
     vote_boost    — multiplier applied
     vwap          — computed VWAP value
     vwap_factor   — confidence multiplier from VWAP alignment
     vol_ratio     — last-bar volume / 20-bar mean
     vol_factor    — confidence multiplier from volume ratio
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
)
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.signals.versions.base import BaseSignalStrategy
from smart_trader.strategy.trend.regime import MarketRegime, TrendState

_DETECTORS: list[tuple[str, callable]] = [
    ("rsi",       detect_rsi),
    ("macd",      detect_macd_cross),
    ("bollinger", detect_bollinger_touch),
    ("ema_bounce",detect_ema_bounce),
]

MIN_RAW_SCORE   = 0.40
# Raised floor for signals with no confirming detector (guards against noise).
# Structural regimes (ACCUMULATION, DISTRIBUTION, *_TRENDING) already act as a
# second layer of confirmation, so the single-signal penalty is waived there.
# In RANGING regimes, mean-reversion setups (RSI/BB) are intrinsically single-
# detector and statistically valid — use a lower floor to avoid over-filtering.
MIN_CONF_SINGLE         = 0.65   # trending/unknown
MIN_CONF_SINGLE_RANGING = 0.58   # ranging — mean-reversion is the primary edge

_RANGING_REGIMES    = frozenset({MarketRegime.BULL_RANGING, MarketRegime.BEAR_RANGING})

# Regimes where the macro structure itself confirms direction — single-detector
# signals are trusted at the caller's min_confidence without extra penalty.
_STRUCTURAL_REGIMES = frozenset([
    MarketRegime.ACCUMULATION,
    MarketRegime.DISTRIBUTION,
    MarketRegime.BULL_TRENDING,
    MarketRegime.BEAR_TRENDING,
])


# ── VWAP ─────────────────────────────────────────────────────────────────────

def _vwap_factor(df: pd.DataFrame, signal_type: str) -> tuple[float, float]:
    """Return (factor, vwap_value) for the latest bar.

    Uses a rolling 24-bar typical-price-weighted VWAP so it never needs a
    session-boundary reset and works on any OHLCV DataFrame.

    Factor table (long signals):
        price / vwap < 0.99   →  1.15   (buying below fair value)
        price / vwap 0.99–1.0 →  1.05
        price / vwap 1.0–1.01 →  0.95
        price / vwap > 1.01   →  0.82   (chasing above fair value)
    Short signals: mirror of the above.
    """
    window  = min(24, len(df))
    recent  = df.iloc[-window:]
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    vwap    = float(
        (typical * recent["volume"]).sum() / (recent["volume"].sum() + 1e-9)
    )
    close   = float(df["close"].iloc[-1])
    ratio   = close / (vwap + 1e-9)

    if signal_type == "long":
        if ratio < 0.99:   factor = 1.15
        elif ratio < 1.00: factor = 1.05
        elif ratio < 1.01: factor = 0.95
        else:              factor = 0.82
    else:                              # short
        if ratio > 1.01:   factor = 1.15
        elif ratio > 1.00: factor = 1.05
        elif ratio > 0.99: factor = 0.95
        else:              factor = 0.82

    return factor, vwap


# ── Volume ratio ──────────────────────────────────────────────────────────────

def _volume_ratio_factor(df: pd.DataFrame) -> tuple[float, float]:
    """Return (factor, ratio) where ratio = last_volume / 20-bar mean.

    Factor table:
        ratio > 2.0  →  1.15   (strong confirmation volume)
        ratio > 1.5  →  1.08
        ratio > 0.7  →  1.00   (normal — neutral)
        ratio > 0.4  →  0.92   (below-average liquidity)
        ratio ≤ 0.4  →  0.85   (very thin volume — penalise)
    """
    vol = df["volume"].astype(float)
    if len(vol) < 21:
        return 1.0, 1.0
    avg   = float(vol.iloc[-21:-1].mean())
    ratio = float(vol.iloc[-1]) / (avg + 1e-9)

    if ratio > 2.0:   factor = 1.15
    elif ratio > 1.5: factor = 1.08
    elif ratio > 0.7: factor = 1.00
    elif ratio > 0.4: factor = 0.92
    else:             factor = 0.85

    return factor, ratio


# ── Voting ────────────────────────────────────────────────────────────────────

_VOTE_BOOST: dict[int, float] = {1: 1.00, 2: 1.20, 3: 1.40}


def _apply_vote_boost(events: list[SignalEvent]) -> list[SignalEvent]:
    """Boost confidence when multiple detectors agree on the same direction.

    Mutates confidence and features in-place; returns the same list.
    """
    by_dir: dict[str, list[SignalEvent]] = {"long": [], "short": []}
    for e in events:
        if e.signal_type in by_dir:
            by_dir[e.signal_type].append(e)

    for direction, sigs in by_dir.items():
        n     = len(sigs)
        boost = _VOTE_BOOST.get(n, 1.40)   # 4+ detectors → same as 3
        for e in sigs:
            e.confidence = round(min(1.0, e.confidence * boost), 4)
            e.features["vote_count"] = float(n)
            e.features["vote_boost"] = boost

    return events


# ── Strategy ──────────────────────────────────────────────────────────────────

class StrategyV2(BaseSignalStrategy):
    """v1 + multi-signal voting + VWAP alignment + continuous volume scoring."""

    VERSION = "v2"

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

        now    = datetime.now(timezone.utc)
        events: list[SignalEvent] = []

        for name, detector in _DETECTORS:
            result: DetectorResult = detector(df)
            if result is None:
                continue

            signal_type, raw_score, features = result
            if raw_score < MIN_RAW_SCORE:
                continue

            # ── base confidence (regime alignment) ───────────────────────
            # vol_spike=False: volume is handled separately below as a
            # continuous multiplier, so the bool path is intentionally off.
            confidence = score_confidence(
                raw_score=raw_score,
                signal_type=signal_type,
                trend_state=trend_state,
                vol_spike=False,
            )
            if confidence <= 0.0:           # hard-blocked by regime
                continue

            # ── VWAP alignment ────────────────────────────────────────────
            vwap_f, vwap_val = _vwap_factor(df, signal_type)
            # In structural regimes (accumulation, distribution, trending)
            # price routinely sits on the "wrong" side of VWAP — penalising
            # it would block valid setups. Allow bonus only, no penalty.
            if trend_state.regime in _STRUCTURAL_REGIMES:
                vwap_f = max(1.0, vwap_f)
            confidence = confidence * vwap_f

            # ── volume ratio ──────────────────────────────────────────────
            vol_f, vol_ratio = _volume_ratio_factor(df)
            confidence       = confidence * vol_f

            confidence = round(min(1.0, confidence), 4)

            # annotate for audit / future ML features
            features["vwap"]       = round(vwap_val, 4)
            features["vwap_factor"]= round(vwap_f,   4)
            features["vol_ratio"]  = round(vol_ratio, 4)
            features["vol_factor"] = round(vol_f,    4)

            events.append(
                SignalEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal_type,
                    source=f"technical_{name}",
                    raw_score=round(raw_score, 4),
                    confidence=confidence,
                    regime=trend_state.regime,
                    strategy_mode=trend_state.strategy_mode,
                    features=features,
                    created_at=now,
                )
            )

        # ── voting boost ──────────────────────────────────────────────────
        events = _apply_vote_boost(events)

        # ── final confidence filter ───────────────────────────────────────
        # Single-detector signals face a raised floor (MIN_CONF_SINGLE) unless
        # the regime is structural — in that case the macro context already
        # acts as a second confirmation so we use the caller's min_confidence.
        is_structural = trend_state.regime in _STRUCTURAL_REGIMES
        is_ranging    = trend_state.regime in _RANGING_REGIMES
        result: list[SignalEvent] = []
        for e in events:
            n = int(e.features.get("vote_count", 1))
            if n == 1 and not is_structural:
                single_floor = MIN_CONF_SINGLE_RANGING if is_ranging else MIN_CONF_SINGLE
                floor = max(min_confidence, single_floor)
            else:
                floor = min_confidence
            if e.confidence >= floor:
                result.append(e)

        result.sort(key=lambda e: e.confidence, reverse=True)
        return result
