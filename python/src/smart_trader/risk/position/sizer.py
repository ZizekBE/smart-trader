"""PositionSizer — Kelly-based position sizing with ATR stop-loss/take-profit.

Strategy-mode parameters:
  ─────────────────────────────────────────────────────────────────
  Mode           ATR mult  Base R:R  Kelly frac  Max pct/trade
  ─────────────────────────────────────────────────────────────────
  conservative   1.5       3:1       0.25×       3 %
  balanced       2.0       3:1       0.25×       5 %
  aggressive     2.5       3:1       0.50×       8 %
  ─────────────────────────────────────────────────────────────────

Dynamic R:R (applied on top of base R:R from mode):
  BULL/BEAR_TRENDING  → × 4/3   (effective 4:1 — let the trend run)
  Ranging / Transition → × 5/6  (effective 2.5:1 — realistic range target)
  HIGH vol            → × 0.80  (tighter TP — harder to reach in whipsaw)
  CRISIS vol          → × 0.70
  LOW  vol            → × 1.20  (wider TP — price moves smoothly)
  Clamp range: [1.5, 6.0]

Kelly fraction (fractional Kelly):
    f* = (p·b − q) / b
where p = signal confidence, q = 1−p, b = reward/risk ratio.
We apply a fraction of full Kelly to limit over-betting.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from smart_trader.risk.models import Portfolio, PositionSize
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode
from smart_trader.strategy.volatility.analyzer import VolatilityState

# Per strategy-mode risk parameters
_MODE_PARAMS: dict[str, dict] = {
    StrategyMode.CONSERVATIVE: {"atr_mult": 1.5, "rr": 3.0, "kelly_frac": 0.25, "max_pct": 0.05},
    StrategyMode.BALANCED:     {"atr_mult": 2.0, "rr": 3.0, "kelly_frac": 0.40, "max_pct": 0.08},
    StrategyMode.AGGRESSIVE:   {"atr_mult": 2.5, "rr": 3.0, "kelly_frac": 0.65, "max_pct": 0.12},
}
_DEFAULT_PARAMS = _MODE_PARAMS[StrategyMode.BALANCED]

# Regimes where we let the trend run (wider TP)
_TRENDING_REGIMES = frozenset({MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING})
# Regimes where price oscillates (tighter TP)
_RANGING_REGIMES  = frozenset({
    MarketRegime.BULL_RANGING, MarketRegime.BEAR_RANGING,
    MarketRegime.DISTRIBUTION, MarketRegime.ACCUMULATION,
})

# Absolute floor: never risk less than this fraction (avoids dust trades)
_MIN_RISK_PCT = 0.001   # 0.1 %


class PositionSizer:
    """Compute position size for a signal given the current portfolio."""

    def __init__(
        self,
        atr_mult_override:   float | None = None,
        rr_override:         float | None = None,
        kelly_frac_override: float | None = None,
    ) -> None:
        self._atr_mult_override   = atr_mult_override
        self._rr_override         = rr_override
        self._kelly_frac_override = kelly_frac_override

    def size(
        self,
        signal:         SignalEvent,
        portfolio:      Portfolio,
        current_price:  float,
        recent_candles: pd.DataFrame | None    = None,
        vol_state:      VolatilityState | None = None,
        hard_cap_pct:   float | None           = None,
        size_scale:     float                  = 1.0,
    ) -> PositionSize:
        params = _MODE_PARAMS.get(signal.strategy_mode, _DEFAULT_PARAMS)

        # ── 1. ATR for dynamic SL/TP ─────────────────────────
        atr = self._atr(recent_candles, current_price) if recent_candles is not None else (
            current_price * 0.015  # fallback: 1.5% of price
        )

        # priority: instance override > vol_state > mode default
        if self._atr_mult_override is not None:
            atr_mult = self._atr_mult_override
        elif vol_state is not None:
            atr_mult = vol_state.atr_mult
        else:
            atr_mult = params["atr_mult"]

        # Dynamic R:R — widen for trending regimes, tighten for ranging + high vol
        base_rr = self._rr_override if self._rr_override is not None else params["rr"]
        rr = _dynamic_rr(signal.regime, vol_state, base_rr)

        if signal.signal_type == "long":
            stop_price        = current_price - atr_mult * atr
            take_profit_price = current_price + rr * atr_mult * atr
        else:  # short
            stop_price        = current_price + atr_mult * atr
            take_profit_price = current_price - rr * atr_mult * atr

        # ── 2. Kelly fraction → notional ─────────────────────
        kelly_scale  = vol_state.kelly_scale if vol_state is not None else 1.0
        kelly        = _kelly_fraction(signal.confidence, rr)
        kf           = self._kelly_frac_override if self._kelly_frac_override is not None else params["kelly_frac"]
        raw_pct      = kelly * kf * kelly_scale
        mode_cap      = params["max_pct"] * max(0.0, min(1.0, size_scale))
        effective_cap = min(mode_cap, hard_cap_pct) if hard_cap_pct is not None else mode_cap
        position_pct  = max(_MIN_RISK_PCT, min(effective_cap, raw_pct))

        # ── 3. Notional and dollar-risk ──────────────────────
        notional      = position_pct * portfolio.total_value
        notional      = min(notional, portfolio.cash)
        quantity      = notional / current_price if current_price > 0 else 0.0

        stop_dist_pct = abs(current_price - stop_price) / current_price
        risk_pct      = (notional * stop_dist_pct) / portfolio.total_value

        return PositionSize(
            symbol=signal.symbol,
            side=signal.signal_type,
            quantity=round(quantity, 8),
            notional=round(notional, 2),
            stop_price=round(stop_price, 2),
            take_profit_price=round(take_profit_price, 2),
            risk_pct=round(risk_pct, 6),
            atr=round(atr, 2),
        )

    @staticmethod
    def _atr(df: pd.DataFrame, fallback_price: float, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return fallback_price * 0.015
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev  = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _dynamic_rr(
    regime:    MarketRegime,
    vol_state: VolatilityState | None,
    base_rr:   float,
) -> float:
    """Adjust reward:risk ratio based on market regime and volatility."""
    if regime in _TRENDING_REGIMES:
        rr = base_rr * (4.0 / 3.0)   # 3 → 4 — let the trend run
    elif regime in _RANGING_REGIMES:
        rr = base_rr * (2.5 / 3.0)   # 3 → 2.5 — realistic range target
    else:
        rr = base_rr

    if vol_state is not None:
        vol_reg = str(vol_state.regime)
        if vol_reg == "crisis":
            rr *= 0.70
        elif vol_reg == "high":
            rr *= 0.80   # tighter TP in high vol (harder to reach)
        elif vol_reg == "low":
            rr *= 1.20   # wider TP in calm markets

    return max(1.5, min(6.0, rr))


def _kelly_fraction(confidence: float, reward_ratio: float) -> float:
    """Full Kelly fraction — caller applies the fractional multiplier."""
    p = max(0.0, min(1.0, confidence))
    q = 1.0 - p
    b = max(0.1, reward_ratio)
    kelly = (p * b - q) / b
    return max(0.0, kelly)
