"""LimitsChecker — validates a proposed position against portfolio limits.

Checks (all must pass for approval):
  1. notional > 0                        — non-zero trade
  2. notional ≤ max_position_size_pct    — single-position cap
  3. (used_margin + notional) ≤ 90 %     — total exposure cap
  4. no duplicate open position (same symbol + side)
"""
from __future__ import annotations

from smart_trader.risk.models import Portfolio, PositionSize

# Hard limit: total open exposure as fraction of portfolio
_MAX_TOTAL_EXPOSURE_PCT = 0.90


class LimitsChecker:
    def __init__(self, max_position_size_pct: float = 0.05) -> None:
        self._max_pos_pct = max_position_size_pct

    def check(
        self,
        portfolio: Portfolio,
        position:  PositionSize,
    ) -> tuple[bool, list[str]]:
        """Return (approved, list_of_failure_reasons).

        An empty reasons list means all checks passed.
        """
        reasons: list[str] = []

        # 1. Notional floor
        if position.notional <= 0:
            reasons.append("notional ≤ 0")

        # 2. Single-position size cap
        # Use a small tolerance to absorb floating-point rounding from the sizer.
        pos_pct = position.notional / portfolio.total_value if portfolio.total_value > 0 else 1.0
        if pos_pct > self._max_pos_pct * 1.0001:  # 0.01% tolerance for float rounding
            reasons.append(
                f"position {pos_pct:.1%} exceeds max {self._max_pos_pct:.1%}"
            )

        # 3. Total exposure cap
        total_exposure = (portfolio.used_margin + position.notional) / portfolio.total_value
        if total_exposure > _MAX_TOTAL_EXPOSURE_PCT:
            reasons.append(
                f"total exposure {total_exposure:.1%} would exceed {_MAX_TOTAL_EXPOSURE_PCT:.0%}"
            )

        # 4. Duplicate position guard
        if position.symbol in portfolio.open_positions:
            existing = portfolio.open_positions[position.symbol]
            if existing.side == position.side:
                reasons.append(
                    f"already holding {position.side} on {position.symbol}"
                )

        return (len(reasons) == 0, reasons)
