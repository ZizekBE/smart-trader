"""Shared data models for the Risk Layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OpenPosition:
    symbol:          str
    side:            str    # 'long' | 'short'
    quantity:        float  # base currency (e.g. BTC)
    entry_price:     float
    current_price:   float

    @property
    def notional(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return self.quantity * (self.current_price - self.entry_price)
        return self.quantity * (self.entry_price - self.current_price)


@dataclass
class Portfolio:
    """Snapshot of current portfolio state passed to every risk check."""
    total_value:     float                          # USDT total (cash + positions)
    cash:            float                          # available USDT
    open_positions:  dict[str, OpenPosition] = field(default_factory=dict)
    peak_value:      float = 0.0                    # all-time high — for drawdown
    daily_pnl:       float = 0.0                    # today's realised + unrealised P&L
    daily_start_value: float = 0.0                  # portfolio value at day open

    def __post_init__(self) -> None:
        if self.peak_value == 0.0:
            self.peak_value = self.total_value
        if self.daily_start_value == 0.0:
            self.daily_start_value = self.total_value

    @property
    def used_margin(self) -> float:
        return sum(p.notional for p in self.open_positions.values())

    @property
    def drawdown_pct(self) -> float:
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.total_value) / self.peak_value)

    @property
    def daily_loss_pct(self) -> float:
        """Positive = loss, negative = gain."""
        if self.daily_start_value <= 0:
            return 0.0
        return max(0.0, (self.daily_start_value - self.total_value) / self.daily_start_value)


@dataclass(slots=True)
class PositionSize:
    symbol:           str
    side:             str    # 'long' | 'short'
    quantity:         float  # base currency amount to trade
    notional:         float  # USDT value
    stop_price:       float
    take_profit_price: float
    risk_pct:         float  # fraction of portfolio risked (e.g. 0.02 = 2%)
    atr:              float  # ATR used for SL/TP calculation


@dataclass(slots=True)
class CircuitBreakerResult:
    tripped:         bool
    reason:          str | None       # 'daily_loss' | 'drawdown' | None
    daily_loss_pct:  float
    drawdown_pct:    float


@dataclass
class RiskDecision:
    approved:      bool
    position_size: PositionSize | None
    reasons:       list[str]          # human-readable explanation
    checks:        dict[str, bool]    # per-check pass/fail

    def __str__(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return f"RiskDecision[{status}] {'; '.join(self.reasons)}"
