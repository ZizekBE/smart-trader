"""RiskManager — single façade that wires all risk checks together.

Usage::
    rm = RiskManager.from_settings()
    decision = rm.evaluate(signal, portfolio, current_price, candles_df)
    if decision.approved:
        place_order(decision.position_size)

Check order (fail-fast):
  1. CircuitBreaker  — hard stop (daily loss / drawdown)
  2. Confidence gate — signal below minimum threshold
  3. PositionSizer   — compute size (may yield 0 → reject)
  4. LimitsChecker   — exposure / duplicate-position guard
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import structlog

from smart_trader.core.settings import get_settings
from smart_trader.risk.circuit_breaker.breaker import CircuitBreaker
from smart_trader.risk.limits.checker import LimitsChecker
from smart_trader.risk.models import Portfolio, RiskDecision
from smart_trader.risk.position.sizer import PositionSizer
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer, VolatilityState

log = structlog.get_logger(__name__)


class RiskManager:
    def __init__(
        self,
        min_confidence:      float        = 0.65,
        max_position_pct:    float        = 0.05,
        max_daily_loss_pct:  float        = 0.02,
        max_drawdown_pct:    float        = 0.10,
        atr_mult:            float | None = None,
        rr_ratio:            float | None = None,
        kelly_frac:          float | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._sizer   = PositionSizer(
            atr_mult_override=atr_mult,
            rr_override=rr_ratio,
            kelly_frac_override=kelly_frac,
        )
        self._limits  = LimitsChecker(max_position_size_pct=max_position_pct)
        self._breaker = CircuitBreaker(
            max_daily_loss_pct=max_daily_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
        )
        self._vol_analyzer = VolatilityAnalyzer()
        self._log = log.bind(component="risk_manager")

    @classmethod
    def from_settings(cls) -> "RiskManager":
        s = get_settings()
        return cls(
            min_confidence=s.confidence_threshold,
            max_position_pct=s.max_position_size_pct,
            max_daily_loss_pct=s.max_daily_loss_pct,
            max_drawdown_pct=s.max_drawdown_pct,
            atr_mult=s.opt_atr_mult,
            rr_ratio=s.opt_rr_ratio,
            kelly_frac=s.opt_kelly_frac,
        )

    # ── one-stop evaluation ────────────────────────────────────

    def evaluate(
        self,
        signal:        SignalEvent,
        portfolio:     Portfolio,
        current_price: float,
        candles_df:    pd.DataFrame | None = None,
        long_df:       pd.DataFrame | None = None,
        size_scale:    float               = 1.0,
    ) -> RiskDecision:
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        # ── 0. volatility regime ──────────────────────────────
        vol_state: Optional[VolatilityState] = None
        if candles_df is not None and long_df is not None:
            try:
                vol_state = self._vol_analyzer.analyse(long_df, candles_df)
                checks["vol_regime"] = not vol_state.skip
                if vol_state.skip:
                    reasons.append(
                        f"vol skip: regime={vol_state.regime} state={vol_state.state}"
                    )
                    self._log.info(
                        "vol_skip", regime=str(vol_state.regime),
                        state=str(vol_state.state), symbol=signal.symbol,
                    )
                    return RiskDecision(approved=False, position_size=None,
                                       reasons=reasons, checks=checks)
            except Exception:
                pass   # vol analysis is best-effort; don't break the pipeline

        # ── 1. circuit breaker ────────────────────────────────
        cb = self._breaker.check(portfolio)
        checks["circuit_breaker"] = not cb.tripped
        if cb.tripped:
            reasons.append(f"circuit_breaker tripped ({cb.reason})")
            self._log.warning("rejected_circuit_breaker", symbol=signal.symbol, reason=cb.reason)
            return RiskDecision(approved=False, position_size=None,
                                reasons=reasons, checks=checks)

        # ── 2. confidence gate ────────────────────────────────
        conf_ok = signal.confidence >= self._min_confidence
        checks["confidence"] = conf_ok
        if not conf_ok:
            reasons.append(
                f"confidence {signal.confidence:.4f} < threshold {self._min_confidence:.4f}"
            )
            self._log.debug("rejected_confidence", symbol=signal.symbol,
                            confidence=signal.confidence, threshold=self._min_confidence)
            return RiskDecision(approved=False, position_size=None,
                                reasons=reasons, checks=checks)

        # ── 3. position sizing ────────────────────────────────
        pos = self._sizer.size(
            signal, portfolio, current_price, candles_df, vol_state,
            hard_cap_pct=self._limits._max_pos_pct,
            size_scale=size_scale,
        )
        checks["position_size"] = pos.notional > 0
        if pos.notional <= 0:
            reasons.append("position size computed to zero (insufficient Kelly or cash)")
            return RiskDecision(approved=False, position_size=None,
                                reasons=reasons, checks=checks)

        # ── 4. limits ─────────────────────────────────────────
        limits_ok, limit_reasons = self._limits.check(portfolio, pos)
        checks["limits"] = limits_ok
        if not limits_ok:
            reasons.extend(limit_reasons)
            self._log.warning("rejected_limits", symbol=signal.symbol, reasons=limit_reasons)
            return RiskDecision(approved=False, position_size=None,
                                reasons=reasons, checks=checks)

        # ── approved ─────────────────────────────────────────
        reasons.append(
            f"approved: qty={pos.quantity:.6f} notional=${pos.notional:.2f} "
            f"SL={pos.stop_price:.2f} TP={pos.take_profit_price:.2f}"
        )
        self._log.info(
            "approved", symbol=signal.symbol, side=pos.side,
            notional=pos.notional, risk_pct=pos.risk_pct,
            confidence=signal.confidence,
        )
        return RiskDecision(approved=True, position_size=pos,
                            reasons=reasons, checks=checks)

    def reset_breaker(self) -> None:
        """Call at start of each trading day."""
        self._breaker.reset()
