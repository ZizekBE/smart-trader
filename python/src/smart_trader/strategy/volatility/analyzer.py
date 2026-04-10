"""VolatilityAnalyzer — two-layer volatility regime detection.

Layer 1 — macro (long timeframe, e.g. 1d):
    Annualised historical volatility + ATR percentile rank
    → VolRegime: LOW / MEDIUM / HIGH / CRISIS

Layer 2 — micro (short timeframe, e.g. 1h):
    Short-term ATR relative to long-term ATR baseline
    → VolState: CONTRACTING / NORMAL / EXPANDING / SPIKE

Usage::
    analyzer = VolatilityAnalyzer()
    state = analyzer.analyse(daily_df, hourly_df)
    print(state.regime, state.state, state.atr_mult, state.kelly_scale)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smart_trader.strategy.volatility.regime import (
    VOL_PARAMS,
    VOL_PREFERRED_SOURCES,
    VolRegime,
    VolState,
)

# ── tuning constants ──────────────────────────────────────────────────────────
# Annualised σ thresholds (crypto-calibrated)
_CRISIS_THRESH = 1.20   # σ > 120 %
_HIGH_THRESH   = 0.70   # σ > 70 %
_MEDIUM_THRESH = 0.40   # σ > 40 %

# vol_ratio thresholds  (short_atr / long_atr_baseline)
_SPIKE_THRESH       = 2.5
_EXPANDING_THRESH   = 1.4
_CONTRACTING_THRESH = 0.7

_HIST_VOL_WINDOW  = 30   # days for short historical-vol estimate
_ATR_RANK_WINDOW  = 252  # bars for ATR percentile rank
_SHORT_ATR_PERIOD = 14   # ATR period for intraday window
_LONG_ATR_PERIOD  = 14   # ATR period for daily baseline


@dataclass(slots=True)
class VolatilityState:
    # ── classification ────────────────────────────────────────
    regime: VolRegime
    state:  VolState

    # ── raw metrics ───────────────────────────────────────────
    hist_vol_annual:  float   # annualised σ from daily returns
    atr_rank:         float   # [0, 1]  — current ATR percentile
    vol_ratio:        float   # short_atr / long_atr_baseline
    long_atr:         float   # daily ATR (absolute)
    short_atr:        float   # intraday ATR (absolute)

    # ── derived execution params ──────────────────────────────
    atr_mult:         float   # SL distance = atr_mult × ATR
    kelly_scale:      float   # position-size multiplier
    skip:             bool    # True → don't open new positions

    # ── preferred signal sources ──────────────────────────────
    preferred_sources: list[str]  # empty = no restriction

    calculated_at: datetime

    def __str__(self) -> str:
        return (
            f"VolatilityState(regime={self.regime} state={self.state} "
            f"σ={self.hist_vol_annual:.0%} atr_rank={self.atr_rank:.0%} "
            f"vol_ratio={self.vol_ratio:.2f} "
            f"atr_mult={self.atr_mult}× kelly_scale={self.kelly_scale} "
            f"skip={self.skip})"
        )


class VolatilityAnalyzer:
    """Two-layer volatility regime classifier.

    Stateless — create once and call analyse() on each bar.
    """

    def analyse(
        self,
        long_df:  pd.DataFrame,   # daily OHLCV
        short_df: pd.DataFrame,   # intraday OHLCV (e.g. 1 h)
    ) -> VolatilityState:
        """Classify volatility regime and state.

        Both DataFrames must have columns: open, high, low, close, volume
        sorted ascending by time (datetime index or 'time' column).
        """
        long_df  = self._ensure_index(long_df)
        short_df = self._ensure_index(short_df)

        # ── Layer 1: macro regime ─────────────────────────────
        hist_vol  = self._hist_vol(long_df)
        atr_rank  = self._atr_rank(long_df)
        long_atr  = self._atr(long_df, _LONG_ATR_PERIOD)
        regime    = self._classify_regime(hist_vol, atr_rank)

        # ── Layer 2: micro state ──────────────────────────────
        short_atr = self._atr(short_df, _SHORT_ATR_PERIOD)
        # normalise intraday ATR by long-term ATR expressed per-bar
        # (daily ATR ÷ 24 gives an hourly baseline)
        hourly_baseline = long_atr / 24.0 if long_atr > 0 else 1e-9
        vol_ratio       = short_atr / hourly_baseline if hourly_baseline > 0 else 1.0
        state           = self._classify_state(vol_ratio)

        params   = VOL_PARAMS[(regime, state)]
        sources  = VOL_PREFERRED_SOURCES[state]

        return VolatilityState(
            regime=regime,
            state=state,
            hist_vol_annual=round(hist_vol, 4),
            atr_rank=round(atr_rank, 4),
            vol_ratio=round(vol_ratio, 4),
            long_atr=round(long_atr, 6),
            short_atr=round(short_atr, 6),
            atr_mult=params["atr_mult"],
            kelly_scale=params["kelly_scale"],
            skip=params["skip"],
            preferred_sources=list(sources),
            calculated_at=datetime.now(timezone.utc),
        )

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        """Exponential ATR — returns scalar for the last bar."""
        if len(df) < period + 1:
            # fallback: average high-low range
            return float((df["high"] - df["low"]).mean())
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev  = close.shift(1)
        tr    = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _hist_vol(df: pd.DataFrame) -> float:
        """Annualised historical volatility from daily log returns."""
        close = df["close"].astype(float)
        n     = min(_HIST_VOL_WINDOW, len(close) - 1)
        if n < 5:
            return 0.5   # fallback: assume medium vol
        log_returns = np.log(close / close.shift(1)).dropna().iloc[-n:]
        daily_std   = float(log_returns.std())
        return daily_std * math.sqrt(252)

    @staticmethod
    def _atr_rank(df: pd.DataFrame) -> float:
        """ATR percentile rank: where is today's ATR vs past N bars? [0, 1]"""
        if len(df) < 15:
            return 0.5
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev  = close.shift(1)
        tr    = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        atr_series = tr.ewm(span=_LONG_ATR_PERIOD, adjust=False).mean()
        window     = atr_series.iloc[-_ATR_RANK_WINDOW:]
        current    = atr_series.iloc[-1]
        return float((window < current).mean())   # fraction of bars below current

    @staticmethod
    def _classify_regime(hist_vol: float, atr_rank: float) -> VolRegime:
        """Classify macro regime.

        Uses hist_vol as primary signal, atr_rank as tie-breaker.
        """
        if hist_vol > _CRISIS_THRESH:
            return VolRegime.CRISIS
        if hist_vol > _HIGH_THRESH:
            return VolRegime.HIGH
        if hist_vol > _MEDIUM_THRESH:
            # borderline: if ATR rank is also high → HIGH
            return VolRegime.HIGH if atr_rank > 0.85 else VolRegime.MEDIUM
        return VolRegime.LOW

    @staticmethod
    def _classify_state(vol_ratio: float) -> VolState:
        if vol_ratio > _SPIKE_THRESH:
            return VolState.SPIKE
        if vol_ratio > _EXPANDING_THRESH:
            return VolState.EXPANDING
        if vol_ratio < _CONTRACTING_THRESH:
            return VolState.CONTRACTING
        return VolState.NORMAL
