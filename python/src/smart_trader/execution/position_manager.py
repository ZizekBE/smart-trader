"""Position Manager — translates agent target positions into order deltas.

Sits between the RL Meta Controller (which outputs abstract target allocations)
and the Order Generator (which turns deltas into concrete exchange orders).
Handles position tracking, delta calculation, and urgency classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class Urgency(StrEnum):
    LOW = "low"        # use limit/TWAP, no rush
    MEDIUM = "medium"  # use TWAP with shorter horizon
    HIGH = "high"      # market order, immediate


@dataclass(slots=True)
class PositionDelta:
    symbol: str
    side: str              # buy | sell
    quantity: float        # absolute
    urgency: Urgency
    order_type: str        # market | limit | twap | vwap
    limit_price: float | None = None
    reason: str = ""


@dataclass(slots=True)
class TrackedPosition:
    symbol: str
    side: str            # long | short | flat
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: datetime = None  # type: ignore[assignment]

    @property
    def notional(self) -> float:
        return self.size * self.entry_price

    @property
    def is_open(self) -> bool:
        return self.size > 1e-12


class PositionManager:
    """Manages position state and computes order deltas from agent actions.

    The agent outputs a target position fraction (e.g. +0.6 = 60% long).
    PositionManager computes the difference between current and target,
    then emits PositionDeltas for the OrderGenerator to execute.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_leverage: float = 10.0,
        deadzone_pct: float = 0.005,     # ignore deltas < 0.5% of portfolio
        urgency_threshold: float = 0.10, # > 10% delta = high urgency
    ) -> None:
        self._capital = initial_capital
        self._max_leverage = max_leverage
        self._deadzone = deadzone_pct
        self._urgency_threshold = urgency_threshold
        self._positions: dict[str, TrackedPosition] = {}
        self._log = log

    def get_position(self, symbol: str) -> TrackedPosition:
        if symbol not in self._positions:
            self._positions[symbol] = TrackedPosition(
                symbol=symbol, side="flat", last_update=datetime.now(timezone.utc),
            )
        return self._positions[symbol]

    def compute_delta(
        self,
        symbol: str,
        target_fraction: float,
        current_price: float,
        risk_budget: float = 0.05,
        portfolio_value: float | None = None,
    ) -> Optional[PositionDelta]:
        """Compute the order delta to reach target_fraction.

        Args:
            symbol:           Trading pair.
            target_fraction:  [-1, 1] — target position as fraction of portfolio.
            current_price:    Current market price.
            risk_budget:      Max risk allowed for this trade.
            portfolio_value:  Override total portfolio value.

        Returns:
            PositionDelta if action needed, None if within dead zone.
        """
        pv = portfolio_value or self._capital
        pos = self.get_position(symbol)

        # current position as fraction of portfolio
        current_notional = 0.0
        if pos.is_open:
            sign = 1.0 if pos.side == "long" else -1.0
            current_notional = sign * pos.size * current_price
        current_fraction = current_notional / (pv + 1e-9)

        diff_fraction = target_fraction - current_fraction
        if abs(diff_fraction) < self._deadzone:
            return None

        target_notional = diff_fraction * pv
        max_notional = pv * risk_budget * self._max_leverage
        clamped_notional = min(abs(target_notional), max_notional)

        side = "buy" if target_notional > 0 else "sell"
        qty = clamped_notional / (current_price + 1e-9)

        urgency = self._classify_urgency(abs(diff_fraction))
        order_type = self._select_order_type(urgency, clamped_notional, pv)

        delta = PositionDelta(
            symbol=symbol,
            side=side,
            quantity=round(qty, 8),
            urgency=urgency,
            order_type=order_type,
            reason=f"target={target_fraction:+.3f} current={current_fraction:+.3f}",
        )
        self._log.info("position_delta", **delta.__dict__)
        return delta

    def update_position(
        self,
        symbol: str,
        side: str,
        filled_qty: float,
        fill_price: float,
    ) -> None:
        """Update tracked position after an order fill."""
        pos = self.get_position(symbol)

        if not pos.is_open:
            pos.side = "long" if side == "buy" else "short"
            pos.size = filled_qty
            pos.entry_price = fill_price
        elif (side == "buy" and pos.side == "long") or (side == "sell" and pos.side == "short"):
            total = pos.size * pos.entry_price + filled_qty * fill_price
            pos.size += filled_qty
            pos.entry_price = total / (pos.size + 1e-12)
        else:
            if filled_qty >= pos.size:
                remaining = filled_qty - pos.size
                if remaining > 1e-12:
                    pos.side = "long" if side == "buy" else "short"
                    pos.size = remaining
                    pos.entry_price = fill_price
                else:
                    pos.side = "flat"
                    pos.size = 0.0
                    pos.entry_price = 0.0
            else:
                pos.size -= filled_qty

        pos.last_update = datetime.now(timezone.utc)

    def update_capital(self, value: float) -> None:
        self._capital = value

    def flatten_all(self, current_prices: dict[str, float]) -> list[PositionDelta]:
        """Generate deltas to close all open positions."""
        deltas = []
        for sym, pos in self._positions.items():
            if pos.is_open:
                side = "sell" if pos.side == "long" else "buy"
                deltas.append(PositionDelta(
                    symbol=sym, side=side, quantity=pos.size,
                    urgency=Urgency.HIGH, order_type="market",
                    reason="flatten_all",
                ))
        return deltas

    def _classify_urgency(self, diff_pct: float) -> Urgency:
        if diff_pct > self._urgency_threshold:
            return Urgency.HIGH
        elif diff_pct > self._urgency_threshold * 0.5:
            return Urgency.MEDIUM
        return Urgency.LOW

    def _select_order_type(self, urgency: Urgency, notional: float, pv: float) -> str:
        participation = notional / (pv + 1e-9)
        if urgency == Urgency.HIGH:
            return "market"
        if participation > 0.05:
            return "twap"
        return "limit"
