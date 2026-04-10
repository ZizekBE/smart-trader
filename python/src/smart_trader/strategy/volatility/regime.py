"""Volatility regime and state enumerations."""
from __future__ import annotations

from enum import StrEnum


class VolRegime(StrEnum):
    """Long-term (macro) volatility level.

    Derived from annualised historical volatility and ATR rank
    over the past 252 trading days.

    Thresholds (annualised σ, crypto-calibrated):
        LOW     < 40 %
        MEDIUM  40 – 70 %
        HIGH    70 – 120 %
        CRISIS  > 120 %
    """
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    CRISIS = "crisis"


class VolState(StrEnum):
    """Short-term (intraday) volatility state.

    Derived from the ratio of recent ATR vs long-term ATR baseline.

    vol_ratio = short_atr / long_atr_baseline
        CONTRACTING  < 0.7   — market coiling, breakout pending
        NORMAL       0.7–1.4 — business as usual
        EXPANDING    1.4–2.5 — directional move in progress
        SPIKE        > 2.5   — extreme intraday range, avoid or min-size
    """
    CONTRACTING = "contracting"
    NORMAL      = "normal"
    EXPANDING   = "expanding"
    SPIKE       = "spike"


# ── derived strategy parameters per (regime, state) ──────────────────────────

# (VolRegime, VolState) → {atr_mult, kelly_scale, skip}
#   atr_mult   : multiplier applied to base ATR for SL distance
#   kelly_scale: fraction of computed Kelly notional to use (1.0 = no change)
#   skip       : True → don't open new positions this bar
VOL_PARAMS: dict[tuple[VolRegime, VolState], dict] = {
    # LOW regime ───────────────────────────────────────────────────────────────
    (VolRegime.LOW, VolState.CONTRACTING): {"atr_mult": 1.5, "kelly_scale": 0.5,  "skip": False},
    (VolRegime.LOW, VolState.NORMAL):      {"atr_mult": 1.5, "kelly_scale": 0.8,  "skip": False},
    (VolRegime.LOW, VolState.EXPANDING):   {"atr_mult": 2.0, "kelly_scale": 1.0,  "skip": False},
    (VolRegime.LOW, VolState.SPIKE):       {"atr_mult": 2.0, "kelly_scale": 0.3,  "skip": False},
    # MEDIUM regime ────────────────────────────────────────────────────────────
    (VolRegime.MEDIUM, VolState.CONTRACTING): {"atr_mult": 1.5, "kelly_scale": 1.0,  "skip": False},
    (VolRegime.MEDIUM, VolState.NORMAL):      {"atr_mult": 2.0, "kelly_scale": 1.0,  "skip": False},
    (VolRegime.MEDIUM, VolState.EXPANDING):   {"atr_mult": 2.0, "kelly_scale": 1.2,  "skip": False},
    (VolRegime.MEDIUM, VolState.SPIKE):       {"atr_mult": 2.5, "kelly_scale": 0.0,  "skip": True},
    # HIGH regime ──────────────────────────────────────────────────────────────
    # Wider stops prevent noise-driven stop-outs; kelly_scale caps position size.
    (VolRegime.HIGH, VolState.CONTRACTING): {"atr_mult": 3.0, "kelly_scale": 0.6,  "skip": False},
    (VolRegime.HIGH, VolState.NORMAL):      {"atr_mult": 3.0, "kelly_scale": 0.7,  "skip": False},
    (VolRegime.HIGH, VolState.EXPANDING):   {"atr_mult": 3.5, "kelly_scale": 0.8,  "skip": False},
    (VolRegime.HIGH, VolState.SPIKE):       {"atr_mult": 3.5, "kelly_scale": 0.0,  "skip": True},
    # CRISIS regime ────────────────────────────────────────────────────────────
    (VolRegime.CRISIS, VolState.CONTRACTING): {"atr_mult": 3.0, "kelly_scale": 0.0, "skip": True},
    (VolRegime.CRISIS, VolState.NORMAL):      {"atr_mult": 3.0, "kelly_scale": 0.0, "skip": True},
    (VolRegime.CRISIS, VolState.EXPANDING):   {"atr_mult": 3.0, "kelly_scale": 0.0, "skip": True},
    (VolRegime.CRISIS, VolState.SPIKE):       {"atr_mult": 3.0, "kelly_scale": 0.0, "skip": True},
}

# Preferred signal sources per VolState
# (empty list means no restriction)
VOL_PREFERRED_SOURCES: dict[VolState, list[str]] = {
    VolState.CONTRACTING: ["rsi", "bollinger"],           # mean reversion
    VolState.NORMAL:      [],                              # all signals OK
    VolState.EXPANDING:   ["macd", "ema_bounce"],          # momentum
    VolState.SPIKE:       ["rsi"],                         # only extreme RSI
}
