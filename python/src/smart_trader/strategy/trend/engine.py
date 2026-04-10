"""TrendEngine — multi-signal weighted-vote market-regime classifier.

Signal weights (sum = 1.0):
  ema_cross   EMA20 vs EMA50   short-term momentum         0.25
  ema_long    EMA50 vs EMA200  macro trend                 0.20
  adx         ADX + DI lines   directional strength        0.25
  obv         OBV slope        volume-confirmed direction  0.15
  structure   HH/HL vs LH/LL   market structure            0.15

Each signal returns a score in [-1, +1]:
  +1 → strongly bullish
  -1 → strongly bearish
   0 → neutral
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from smart_trader.strategy.trend.indicators import (
    adx as calc_adx,
    ema,
    linear_slope,
    obv as calc_obv,
)
from smart_trader.strategy.trend.regime import (
    REGIME_TO_MODE,
    MarketRegime,
    TrendState,
)

# ── signal weights ──────────────────────────────────────────────
_WEIGHTS: dict[str, float] = {
    "ema_cross": 0.25,
    "ema_long":  0.20,
    "adx":       0.25,
    "obv":       0.15,
    "structure": 0.15,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"

# ── tuning constants ────────────────────────────────────────────
_ADX_TRENDING   = 25.0   # ADX above this → trending market
_ADX_WEAK       = 15.0   # ADX below this → directionless
_EMA_SCALE      = 0.02   # ±2% EMA divergence maps to ±1 signal
_EMA_MACRO_SCALE= 0.05   # ±5% for macro EMA divergence
_OBV_SCALE      = 0.01   # relative OBV slope scale
_STRUCT_WINDOW  = 20     # bars used for market-structure detection
_OBV_WINDOW     = 20     # bars used for OBV slope calculation
_REGIME_THRESH  = 0.35   # |direction| above this → trending regime


class TrendEngine:
    """Stateless market-regime analyser.

    Usage::
        engine = TrendEngine()
        state  = engine.analyse(df, symbol="BTC/USDT", timeframe="1d")
    """

    MIN_CANDLES = 60  # minimum rows for reliable signals

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> TrendState:
        """Classify market regime from an OHLCV DataFrame.

        `df` must contain columns: time, open, high, low, close, volume
        sorted ascending by time.
        """
        if len(df) < self.MIN_CANDLES:
            raise ValueError(
                f"TrendEngine needs ≥ {self.MIN_CANDLES} candles, got {len(df)}"
            )

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── individual signal scores ──────────────────────────────
        signals: dict[str, float] = {
            "ema_cross": self._ema_cross_signal(close),
            "ema_long":  self._ema_long_signal(close),
            "adx":       self._adx_signal(high, low, close),
            "obv":       self._obv_signal(close, volume),
            "structure": self._structure_signal(high, low),
        }

        # ── weighted vote → aggregate direction [-1, +1] ──────────
        direction = sum(_WEIGHTS[k] * v for k, v in signals.items())
        direction = max(-1.0, min(1.0, direction))
        strength  = abs(direction)

        # ── regime classification (needs raw ADX for trending flag) ─
        adx_val, di_plus, di_minus = self._raw_adx(high, low, close)
        regime        = self._classify_regime(direction, adx_val, volume)
        strategy_mode = REGIME_TO_MODE[regime]

        return TrendState(
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            strategy_mode=strategy_mode,
            direction=round(direction, 4),
            strength=round(strength, 4),
            signals={k: round(v, 4) for k, v in signals.items()},
            calculated_at=datetime.now(timezone.utc),
        )

    # ── signal implementations ─────────────────────────────────────

    def _ema_cross_signal(self, close: pd.Series, fast: int = 20, slow: int = 50) -> float:
        """EMA20 vs EMA50 — relative distance scaled to [-1, 1]."""
        if len(close) < slow:
            return 0.0
        e_fast = ema(close, fast).iloc[-1]
        e_slow = ema(close, slow).iloc[-1]
        rel = (e_fast - e_slow) / (e_slow + 1e-9)
        return max(-1.0, min(1.0, rel / _EMA_SCALE))

    def _ema_long_signal(self, close: pd.Series, slow: int = 50, vlong: int = 200) -> float:
        """EMA50 vs EMA200 macro trend.  Falls back to price vs EMA50 if short history."""
        if len(close) < vlong:
            if len(close) < slow:
                return 0.0
            e = ema(close, slow).iloc[-1]
            rel = (close.iloc[-1] - e) / (e + 1e-9)
            return max(-1.0, min(1.0, rel / _EMA_MACRO_SCALE))
        e_slow  = ema(close, slow).iloc[-1]
        e_vlong = ema(close, vlong).iloc[-1]
        rel = (e_slow - e_vlong) / (e_vlong + 1e-9)
        return max(-1.0, min(1.0, rel / _EMA_MACRO_SCALE))

    def _raw_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple[float, float, float]:
        adx_line, di_plus, di_minus = calc_adx(high, low, close, period)
        return adx_line.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]

    def _adx_signal(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """ADX directional score: sign from DI lines, magnitude from ADX strength."""
        adx_val, di_plus, di_minus = self._raw_adx(high, low, close)
        if math.isnan(adx_val) or adx_val < _ADX_WEAK:
            return 0.0
        direction = 1.0 if di_plus > di_minus else -1.0
        strength  = min(1.0, (adx_val - _ADX_WEAK) / (50.0 - _ADX_WEAK))
        return direction * strength

    def _obv_signal(self, close: pd.Series, volume: pd.Series) -> float:
        """OBV slope over last N bars, scaled to [-1, 1]."""
        obv_series   = calc_obv(close, volume)
        slope_series = linear_slope(obv_series, _OBV_WINDOW)
        slope = slope_series.iloc[-1]
        if math.isnan(slope):
            return 0.0
        return max(-1.0, min(1.0, slope / _OBV_SCALE))

    def _structure_signal(self, high: pd.Series, low: pd.Series) -> float:
        """Market structure: count (HH + HL) vs (LH + LL) over the last N bars."""
        window = min(_STRUCT_WINDOW, len(high))
        h = high.iloc[-window:]
        l = low.iloc[-window:]

        hh = sum(h.iloc[i] > h.iloc[i - 1] for i in range(1, len(h)))
        hl = sum(l.iloc[i] > l.iloc[i - 1] for i in range(1, len(l)))
        lh = sum(h.iloc[i] < h.iloc[i - 1] for i in range(1, len(h)))
        ll = sum(l.iloc[i] < l.iloc[i - 1] for i in range(1, len(l)))

        total = hh + hl + lh + ll
        if total == 0:
            return 0.0
        return (hh + hl - lh - ll) / total  # [-1, 1]

    # ── regime classification ──────────────────────────────────────

    def _classify_regime(
        self,
        direction: float,
        adx_val: float,
        volume: pd.Series,
    ) -> MarketRegime:
        trending = not math.isnan(adx_val) and adx_val >= _ADX_TRENDING

        if direction > _REGIME_THRESH:
            return MarketRegime.BULL_TRENDING if trending else MarketRegime.BULL_RANGING

        if direction < -_REGIME_THRESH:
            return MarketRegime.BEAR_TRENDING if trending else MarketRegime.BEAR_RANGING

        # sideways zone — distinguish accumulation vs distribution via volume slope
        vol_slope = linear_slope(volume, _OBV_WINDOW).iloc[-1]
        if math.isnan(vol_slope) or abs(vol_slope) < 0.001:
            return MarketRegime.BULL_RANGING if direction >= 0 else MarketRegime.BEAR_RANGING

        return MarketRegime.ACCUMULATION if vol_slope > 0 else MarketRegime.DISTRIBUTION
