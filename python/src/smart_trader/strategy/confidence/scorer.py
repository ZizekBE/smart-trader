"""ConfidenceScorer — adjusts a detector's raw score using market context.

Factors applied in order:
  1. Hard regime block  — counter-trend signals in strong trending markets → 0
  2. Regime alignment   — signal direction vs macro trend (+20 % / −60 %)
  3. Volume confirmation — abnormal volume lifts confidence (+10 %)
  4. Hard cap at 1.0

Hard block logic:
  In BULL_TRENDING (strength > BLOCK_STRENGTH_THRESH): short signals blocked.
  In BEAR_TRENDING (strength > BLOCK_STRENGTH_THRESH): long  signals blocked.
  In ACCUMULATION  (strength > BLOCK_ACCUM_THRESH):    short signals blocked
      (smart money is building longs; shorting into accumulation is wrong).
  In DISTRIBUTION  (strength > BLOCK_ACCUM_THRESH):    long  signals blocked
      (smart money is distributing; buying into distribution is wrong).
"""
from __future__ import annotations

from smart_trader.strategy.trend.regime import MarketRegime, TrendState

# Regime-alignment multipliers for LONG signals
_LONG_REGIME_FACTOR: dict[MarketRegime, float] = {
    MarketRegime.BULL_TRENDING: 1.25,
    MarketRegime.BULL_RANGING:  1.10,
    MarketRegime.ACCUMULATION:  1.05,
    MarketRegime.DISTRIBUTION:  0.70,
    MarketRegime.BEAR_RANGING:  0.50,
    MarketRegime.BEAR_TRENDING: 0.30,
}

# Regime-alignment multipliers for SHORT signals (mirror of long)
_SHORT_REGIME_FACTOR: dict[MarketRegime, float] = {
    MarketRegime.BEAR_TRENDING: 1.25,
    MarketRegime.BEAR_RANGING:  1.10,
    MarketRegime.DISTRIBUTION:  1.05,
    MarketRegime.ACCUMULATION:  0.70,
    MarketRegime.BULL_RANGING:  0.50,
    MarketRegime.BULL_TRENDING: 0.30,
}

_VOL_BOOST = 1.10

# When trend strength exceeds this threshold in a trending regime,
# counter-direction signals are hard-blocked (score → 0).
_BLOCK_STRENGTH_THRESH = 0.30

# Softer block threshold for accumulation/distribution regimes.
# Smart money is clearly positioned; opposing them is high-risk.
_BLOCK_ACCUM_THRESH = 0.25


def score_confidence(
    raw_score: float,
    signal_type: str,
    trend_state: TrendState,
    vol_spike: bool = False,
) -> float:
    """Return a context-adjusted confidence score in [0, 1].

    Args:
        raw_score:    Detector output in [0, 1].
        signal_type:  'long' or 'short'.
        trend_state:  Current TrendState from TrendEngine.
        vol_spike:    Whether the last bar had abnormal volume.
    """
    regime   = trend_state.regime
    strength = trend_state.strength

    # ── 1. hard block counter-trend signals in strong trends ───
    if strength >= _BLOCK_STRENGTH_THRESH:
        if regime == MarketRegime.BULL_TRENDING and signal_type == "short":
            return 0.0
        if regime == MarketRegime.BEAR_TRENDING and signal_type == "long":
            return 0.0

    # Block shorts in accumulation and longs in distribution
    if strength >= _BLOCK_ACCUM_THRESH:
        if regime == MarketRegime.ACCUMULATION and signal_type == "short":
            return 0.0
        if regime == MarketRegime.DISTRIBUTION and signal_type == "long":
            return 0.0

    # ── 2. regime alignment ────────────────────────────────────
    table       = _LONG_REGIME_FACTOR if signal_type == "long" else _SHORT_REGIME_FACTOR
    base_factor = table[regime]
    # interpolate between 1.0 and the table value using trend strength
    regime_factor = 1.0 + (base_factor - 1.0) * strength

    confidence = raw_score * regime_factor

    # ── 3. volume confirmation ─────────────────────────────────
    if vol_spike:
        confidence *= _VOL_BOOST

    return min(1.0, confidence)
