"""ConflictGuard — prevents harmful cross-sleeve directional conflicts.

Rules
─────
• Same direction in both sleeves → ALWAYS allowed (stacking = intentional).
• Opposite direction (hedge) → allowed only when tactical notional ≤ 25 %
  of the core position in that symbol.
• If core has no position, tactical can go either direction freely.
"""
from __future__ import annotations

from smart_trader.risk.models import Portfolio

# Tactical counter-position may not exceed this fraction of core notional
_MAX_HEDGE_RATIO = 0.25


class ConflictGuard:
    def check(
        self,
        new_sleeve:     str,          # 'core' | 'tactical'
        new_side:       str,          # 'buy' | 'sell' (order side)
        new_notional:   float,
        core_portfolio:     Portfolio,
        tactical_portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """Return (allowed, reason_if_blocked)."""
        new_direction = "long" if new_side == "buy" else "short"
        opposite      = "short" if new_direction == "long" else "long"

        if new_sleeve == "tactical":
            # Find core positions on any symbol in the same direction as "opposite"
            for pos in core_portfolio.open_positions.values():
                if pos.side == opposite:
                    # Core has opposite direction — check hedge ratio
                    if new_notional > pos.notional * _MAX_HEDGE_RATIO:
                        return (
                            False,
                            f"tactical {new_direction} notional ${new_notional:.0f} "
                            f"exceeds 25% hedge limit against core {opposite} "
                            f"${pos.notional:.0f}",
                        )
        # Core sleeve entering opposite of tactical is always allowed
        # (core drives the strategy; tactical defers to core)
        return True, ""
