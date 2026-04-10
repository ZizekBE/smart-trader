"""CapitalAllocator — tracks sleeve budget and available cash.

Each sleeve is given a fixed fraction of the initial portfolio value.
Budget is *static* (based on initial capital) so it doesn't shrink during
drawdowns, which would create a compounding problem during recovery.

Available cash = sleeve_budget - notional of sleeve's open positions.
"""
from __future__ import annotations


class CapitalAllocator:
    """Compute per-sleeve cash budgets.

    Args:
        initial_capital:   Total starting portfolio value (USDT).
        long_budget_pct:   Fraction reserved for core sleeve (e.g. 0.60).
        short_budget_pct:  Fraction reserved for tactical sleeve (e.g. 0.30).
    """

    def __init__(
        self,
        initial_capital:  float,
        long_budget_pct:  float = 0.60,
        short_budget_pct: float = 0.30,
    ) -> None:
        self._initial = initial_capital
        self._long_budget  = initial_capital * long_budget_pct
        self._short_budget = initial_capital * short_budget_pct

    @property
    def long_budget(self) -> float:
        return self._long_budget

    @property
    def short_budget(self) -> float:
        return self._short_budget

    def sleeve_budget(self, sleeve: str) -> float:
        return self._long_budget if sleeve == "core" else self._short_budget
