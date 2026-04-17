"""Phase 2.2 — Regime-aware adaptive parameter table.

Maps the combination of MarketRegime (trend) + VolRegime (volatility) to a set
of parameter overrides that the trading loop applies each tick.  Extends the
existing VOL_PARAMS pattern (volatility/regime.py) with TrendState awareness.

Lookup logic
────────────
1. If vol_state.skip is True → AdaptiveParams.skip = True (CRISIS guard is
   already enforced by VOL_PARAMS; we simply propagate it here).
2. Otherwise look up (MarketRegime, VolRegime) in TREND_VOL_PARAMS.
3. Fall back to DEFAULT_PARAMS if the combo is missing.

Design principles
─────────────────
- TRENDING regimes: larger R:R already handled by PositionSizer internally;
  here we allow lower min_confidence (engine already found a good signal).
- RANGING regimes: require ≥ 2 confirming detectors (min_votes=2) and a
  higher confidence threshold to avoid noise trades.
- DISTRIBUTION: defensive — halve kelly_scale, require 2 votes, raise conf.
- CRISIS vol: skip=True, propagated from vol_state.

Usage::
    from smart_trader.strategy.adaptive_params import RegimeParamAdapter

    adapter = RegimeParamAdapter()
    params  = adapter.get(trend_state, vol_state)
    if params.skip:
        return          # skip the entire tick
    events = signal_engine.analyse(..., min_confidence=params.min_confidence,
                                        min_votes=params.min_votes)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smart_trader.strategy.trend.regime import MarketRegime

if TYPE_CHECKING:
    from smart_trader.strategy.trend.regime import TrendState
    from smart_trader.strategy.volatility.regime import VolatilityState


@dataclass(frozen=True)
class AdaptiveParams:
    min_confidence: float   # floor passed to SignalEngine.analyse()
    kelly_scale:    float   # multiplier on top of vol-regime kelly_scale
    min_votes:      int     # Phase-1.2 multi-source gate
    skip:           bool    # if True, skip the entire tick
    label:          str     # human-readable description for logging

    def __str__(self) -> str:
        return (
            f"{self.label}  conf≥{self.min_confidence:.2f}  "
            f"kelly×{self.kelly_scale:.1f}  votes≥{self.min_votes}"
            f"{'  SKIP' if self.skip else ''}"
        )


# ── Default (fallback) ────────────────────────────────────────────────────────
DEFAULT_PARAMS = AdaptiveParams(
    min_confidence=0.60,
    kelly_scale=0.8,
    min_votes=1,
    skip=False,
    label="default",
)

SKIP_PARAMS = AdaptiveParams(
    min_confidence=1.0,
    kelly_scale=0.0,
    min_votes=999,
    skip=True,
    label="skip",
)

# ── Regime table (MarketRegime, VolRegime) → AdaptiveParams ──────────────────
# Only the vol regime string is used (LOW / MEDIUM / HIGH / CRISIS) so we can
# match without importing the enum here — callers pass the string value.
#
# Strategy intent per row:
#  BULL_TRENDING        — ride the trend, accept slightly weaker signals
#  BEAR_TRENDING        — ride the short trend, same relaxed threshold
#  BULL/BEAR_RANGING    — mean-reversion noise is high; require 2 votes
#  ACCUMULATION         — potential breakout; balanced, but 2 votes for safety
#  DISTRIBUTION         — defensive: half kelly, 2 votes, higher conf
# ─────────────────────────────────────────────────────────────────────────────
_LOW    = "low"
_MED    = "medium"
_HIGH   = "high"
_CRISIS = "crisis"

TREND_VOL_PARAMS: dict[tuple[str, str], AdaptiveParams] = {

    # ── BULL_TRENDING ─────────────────────────────────────────────────────────
    (MarketRegime.BULL_TRENDING, _LOW):    AdaptiveParams(0.55, 1.0, 1, False, "bull_trend/low_vol"),
    (MarketRegime.BULL_TRENDING, _MED):   AdaptiveParams(0.58, 1.0, 1, False, "bull_trend/med_vol"),
    (MarketRegime.BULL_TRENDING, _HIGH):  AdaptiveParams(0.65, 0.7, 1, False, "bull_trend/high_vol"),
    (MarketRegime.BULL_TRENDING, _CRISIS):SKIP_PARAMS,

    # ── BEAR_TRENDING ─────────────────────────────────────────────────────────
    (MarketRegime.BEAR_TRENDING, _LOW):   AdaptiveParams(0.55, 1.0, 1, False, "bear_trend/low_vol"),
    (MarketRegime.BEAR_TRENDING, _MED):   AdaptiveParams(0.58, 1.0, 1, False, "bear_trend/med_vol"),
    (MarketRegime.BEAR_TRENDING, _HIGH):  AdaptiveParams(0.65, 0.7, 1, False, "bear_trend/high_vol"),
    (MarketRegime.BEAR_TRENDING, _CRISIS):SKIP_PARAMS,

    # ── BULL_RANGING ──────────────────────────────────────────────────────────
    (MarketRegime.BULL_RANGING, _LOW):    AdaptiveParams(0.62, 0.8, 2, False, "bull_range/low_vol"),
    (MarketRegime.BULL_RANGING, _MED):    AdaptiveParams(0.65, 0.7, 2, False, "bull_range/med_vol"),
    (MarketRegime.BULL_RANGING, _HIGH):   AdaptiveParams(0.72, 0.5, 2, False, "bull_range/high_vol"),
    (MarketRegime.BULL_RANGING, _CRISIS): SKIP_PARAMS,

    # ── BEAR_RANGING ──────────────────────────────────────────────────────────
    (MarketRegime.BEAR_RANGING, _LOW):    AdaptiveParams(0.62, 0.8, 2, False, "bear_range/low_vol"),
    (MarketRegime.BEAR_RANGING, _MED):    AdaptiveParams(0.65, 0.7, 2, False, "bear_range/med_vol"),
    (MarketRegime.BEAR_RANGING, _HIGH):   AdaptiveParams(0.72, 0.5, 2, False, "bear_range/high_vol"),
    (MarketRegime.BEAR_RANGING, _CRISIS): SKIP_PARAMS,

    # ── ACCUMULATION ──────────────────────────────────────────────────────────
    (MarketRegime.ACCUMULATION, _LOW):    AdaptiveParams(0.58, 0.9, 2, False, "accumulation/low_vol"),
    (MarketRegime.ACCUMULATION, _MED):    AdaptiveParams(0.60, 0.8, 2, False, "accumulation/med_vol"),
    (MarketRegime.ACCUMULATION, _HIGH):   AdaptiveParams(0.68, 0.6, 2, False, "accumulation/high_vol"),
    (MarketRegime.ACCUMULATION, _CRISIS): SKIP_PARAMS,

    # ── DISTRIBUTION — defensive ──────────────────────────────────────────────
    (MarketRegime.DISTRIBUTION, _LOW):    AdaptiveParams(0.65, 0.5, 2, False, "distribution/low_vol"),
    (MarketRegime.DISTRIBUTION, _MED):    AdaptiveParams(0.70, 0.5, 2, False, "distribution/med_vol"),
    (MarketRegime.DISTRIBUTION, _HIGH):   AdaptiveParams(0.75, 0.3, 2, False, "distribution/high_vol"),
    (MarketRegime.DISTRIBUTION, _CRISIS): SKIP_PARAMS,
}


class RegimeParamAdapter:
    """Stateless lookup helper; one instance can be shared across ticks."""

    def get(
        self,
        trend_state: "TrendState | None",
        vol_state:   "VolatilityState | None",
    ) -> AdaptiveParams:
        # CRISIS skip propagated from vol layer
        if vol_state is not None and vol_state.skip:
            return SKIP_PARAMS

        if trend_state is None or vol_state is None:
            return DEFAULT_PARAMS

        regime   = trend_state.regime
        vol_key  = str(vol_state.regime)   # "low" / "medium" / "high" / "crisis"

        return TREND_VOL_PARAMS.get((regime, vol_key), DEFAULT_PARAMS)
