"""Risk Guard — hard safety limits that cannot be overridden by the RL agent.

This is the last line of defense before orders hit the exchange.
All checks are deterministic and conservative.  If any check fails,
the order is rejected outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from smart_trader.execution.position_manager import PositionDelta, TrackedPosition

log = structlog.get_logger(__name__)


@dataclass
class RiskLimits:
    max_position_notional: float = 50_000.0    # per-symbol cap in USD
    max_single_order_notional: float = 10_000.0
    max_total_exposure: float = 100_000.0       # across all positions
    max_leverage: float = 10.0
    max_daily_loss_pct: float = 0.03            # 3% portfolio
    max_drawdown_pct: float = 0.10              # 10% from peak
    min_margin_ratio: float = 0.02              # 2% maintenance margin
    max_orders_per_minute: int = 10
    max_concentration_pct: float = 0.40         # max 40% in one asset
    cooldown_after_loss_s: int = 300            # 5 min cooldown after big loss


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    daily_loss: float = 0.0
    peak_value: float = 0.0
    current_value: float = 0.0
    orders_this_minute: int = 0
    last_minute_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_big_loss_time: Optional[datetime] = None
    halted: bool = False
    halt_reason: str = ""


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""
    adjusted_delta: Optional[PositionDelta] = None


class RiskGuard:
    """Enforces hard risk limits on all outgoing orders.

    Every PositionDelta must pass through check() before execution.
    The guard can reject, reduce, or pass through deltas.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            peak_value=initial_capital,
            current_value=initial_capital,
        )
        self._positions: dict[str, TrackedPosition] = {}
        self._log = log

    def check(
        self,
        delta: PositionDelta,
        current_price: float,
        portfolio_value: float,
    ) -> RiskCheckResult:
        """Validate a PositionDelta against all risk limits.

        Returns RiskCheckResult with allowed=True if the order can proceed,
        or allowed=False with the rejection reason.
        """
        if self.state.halted:
            return RiskCheckResult(False, f"trading halted: {self.state.halt_reason}")

        # --- single order size ---
        order_notional = delta.quantity * current_price
        if order_notional > self.limits.max_single_order_notional:
            reduced_qty = self.limits.max_single_order_notional / (current_price + 1e-9)
            self._log.warning("order_capped", original=delta.quantity, reduced=reduced_qty)
            delta = PositionDelta(
                symbol=delta.symbol, side=delta.side, quantity=round(reduced_qty, 8),
                urgency=delta.urgency, order_type=delta.order_type,
                reason=f"{delta.reason} [capped by risk guard]",
            )
            order_notional = reduced_qty * current_price

        # --- position concentration ---
        total_exposure = self._total_exposure(current_price)
        new_exposure = total_exposure + order_notional
        if new_exposure > self.limits.max_total_exposure:
            return RiskCheckResult(False, f"total exposure {new_exposure:.0f} > limit {self.limits.max_total_exposure:.0f}")

        symbol_exposure = self._symbol_exposure(delta.symbol, current_price) + order_notional
        if symbol_exposure / (portfolio_value + 1e-9) > self.limits.max_concentration_pct:
            return RiskCheckResult(False, f"concentration {delta.symbol} exceeds {self.limits.max_concentration_pct:.0%}")

        # --- daily loss limit ---
        if self.state.daily_loss / (portfolio_value + 1e-9) > self.limits.max_daily_loss_pct:
            self._halt(f"daily loss {self.state.daily_loss:.2f} exceeds limit")
            return RiskCheckResult(False, self.state.halt_reason)

        # --- drawdown limit ---
        dd = (self.state.peak_value - self.state.current_value) / (self.state.peak_value + 1e-9)
        if dd > self.limits.max_drawdown_pct:
            self._halt(f"drawdown {dd:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}")
            return RiskCheckResult(False, self.state.halt_reason)

        # --- rate limit ---
        now = datetime.now(timezone.utc)
        if (now - self.state.last_minute_start).total_seconds() > 60:
            self.state.orders_this_minute = 0
            self.state.last_minute_start = now
        if self.state.orders_this_minute >= self.limits.max_orders_per_minute:
            return RiskCheckResult(False, "rate limit exceeded")

        # --- loss cooldown ---
        if self.state.last_big_loss_time:
            elapsed = (now - self.state.last_big_loss_time).total_seconds()
            if elapsed < self.limits.cooldown_after_loss_s:
                return RiskCheckResult(False, f"cooldown active ({elapsed:.0f}s / {self.limits.cooldown_after_loss_s}s)")

        self.state.orders_this_minute += 1
        return RiskCheckResult(True, adjusted_delta=delta)

    def record_trade_result(self, pnl: float, portfolio_value: float) -> None:
        """Update risk state after a trade settles."""
        self.state.daily_pnl += pnl
        if pnl < 0:
            self.state.daily_loss += abs(pnl)
            if abs(pnl) > portfolio_value * 0.01:
                self.state.last_big_loss_time = datetime.now(timezone.utc)

        self.state.current_value = portfolio_value
        self.state.peak_value = max(self.state.peak_value, portfolio_value)

    def reset_daily(self) -> None:
        """Reset daily counters (call at UTC midnight)."""
        self.state.daily_pnl = 0.0
        self.state.daily_loss = 0.0
        self.state.orders_this_minute = 0
        if self.state.halted and "daily" in self.state.halt_reason:
            self.state.halted = False
            self.state.halt_reason = ""
            self._log.info("risk_halt_lifted")

    def force_resume(self) -> None:
        """Manual override to resume trading after a halt."""
        self.state.halted = False
        self.state.halt_reason = ""
        self._log.warning("risk_halt_force_resumed")

    def update_positions(self, positions: dict[str, TrackedPosition]) -> None:
        self._positions = positions

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self._log.critical("trading_halted", reason=reason)

    def _total_exposure(self, ref_price: float) -> float:
        return sum(p.notional for p in self._positions.values() if p.is_open)

    def _symbol_exposure(self, symbol: str, price: float) -> float:
        pos = self._positions.get(symbol)
        return pos.notional if pos and pos.is_open else 0.0
