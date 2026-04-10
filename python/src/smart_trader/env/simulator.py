"""Market simulator — realistic execution modeling for RL training.

Models slippage, fees, liquidity impact, and latency for both spot and
futures.  Replays historical OHLCV data and simulates order fills with
configurable fidelity levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class FillModel(StrEnum):
    SIMPLE = "simple"       # fill at close ± fixed slippage
    BAR_RANGE = "bar_range" # fill within [low, high] based on order size
    DEPTH = "depth"         # liquidity-aware fill using depth profile


@dataclass
class SimulatorConfig:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    fill_model: FillModel = FillModel.BAR_RANGE
    slippage_bps: float = 5.0         # base slippage in bps (simple model)
    depth_profile_usd: float = 500_000  # avg depth within 1% of mid
    latency_ms: float = 10.0          # simulated decision-to-fill delay
    funding_enabled: bool = True
    max_leverage: float = 10.0
    maintenance_margin_pct: float = 0.005
    initial_margin_pct: float = 0.01


@dataclass
class Position:
    symbol: str
    side: str            # long | short | flat
    size: float = 0.0    # absolute quantity
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0

    @property
    def notional(self) -> float:
        return self.size * self.entry_price

    @property
    def is_open(self) -> bool:
        return self.size > 1e-12


@dataclass
class FillResult:
    fill_price: float
    fill_qty: float
    fee: float
    slippage_cost: float
    market_impact: float
    status: str  # filled | partial | rejected


class MarketSimulator:
    """Stateful market simulator operating on OHLCV bar data."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.cfg = config or SimulatorConfig()
        self._rng = np.random.default_rng(42)

    def simulate_fill(
        self,
        side: str,          # buy | sell
        quantity: float,
        bar: dict,          # {open, high, low, close, volume}
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> FillResult:
        """Simulate order fill against a historical bar."""
        if self.cfg.fill_model == FillModel.SIMPLE:
            return self._simple_fill(side, quantity, bar)
        elif self.cfg.fill_model == FillModel.BAR_RANGE:
            return self._bar_range_fill(side, quantity, bar, order_type, limit_price)
        else:
            return self._depth_fill(side, quantity, bar)

    def calculate_funding(
        self,
        position: Position,
        funding_rate: float,
        mark_price: float,
    ) -> float:
        """Calculate funding payment for a futures position."""
        if not self.cfg.funding_enabled or not position.is_open:
            return 0.0
        notional = position.size * mark_price
        sign = 1.0 if position.side == "long" else -1.0
        return -sign * notional * funding_rate

    def check_liquidation(
        self,
        position: Position,
        mark_price: float,
    ) -> bool:
        """Check if position would be liquidated at current mark price."""
        if not position.is_open or position.leverage <= 1.0:
            return False
        pnl_pct = self._position_pnl_pct(position, mark_price)
        margin_pct = 1.0 / position.leverage
        return pnl_pct <= -(margin_pct - self.cfg.maintenance_margin_pct)

    def estimate_market_impact(
        self,
        quantity: float,
        price: float,
        volume: float,
    ) -> float:
        """Estimate non-linear market impact as a fraction of price.

        Uses a square-root model: impact ∝ sqrt(order_size / avg_volume).
        """
        order_notional = quantity * price
        participation = order_notional / (self.cfg.depth_profile_usd + 1e-9)
        return 0.0001 * np.sqrt(max(participation, 0))

    # ── fill models ────────────────────────────────────────────

    def _simple_fill(self, side: str, qty: float, bar: dict) -> FillResult:
        price = bar["close"]
        slip_frac = self.cfg.slippage_bps / 10_000
        fill = price * (1 + slip_frac) if side == "buy" else price * (1 - slip_frac)
        fee = fill * qty * self.cfg.taker_fee
        return FillResult(
            fill_price=fill, fill_qty=qty, fee=fee,
            slippage_cost=abs(fill - price) * qty, market_impact=0.0,
            status="filled",
        )

    def _bar_range_fill(
        self, side: str, qty: float, bar: dict,
        order_type: str, limit_price: float | None,
    ) -> FillResult:
        o, h, l, c, v = bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]

        if order_type == "limit" and limit_price is not None:
            if side == "buy" and limit_price < l:
                return FillResult(0, 0, 0, 0, 0, "rejected")
            if side == "sell" and limit_price > h:
                return FillResult(0, 0, 0, 0, 0, "rejected")
            fill = limit_price
            fee_rate = self.cfg.maker_fee
        else:
            impact = self.estimate_market_impact(qty, c, v)
            noise = self._rng.uniform(-0.2, 0.8)
            bar_range = h - l if h > l else c * 0.001
            slip = (impact + noise * (bar_range / c * 0.1))
            fill = c * (1 + slip) if side == "buy" else c * (1 - slip)
            fill = np.clip(fill, l, h)
            fee_rate = self.cfg.taker_fee

        fee = fill * qty * fee_rate
        slip_cost = abs(fill - c) * qty
        return FillResult(
            fill_price=fill, fill_qty=qty, fee=fee,
            slippage_cost=slip_cost,
            market_impact=self.estimate_market_impact(qty, c, v) * c * qty,
            status="filled",
        )

    def _depth_fill(self, side: str, qty: float, bar: dict) -> FillResult:
        """Liquidity-aware fill using synthetic depth profile."""
        c, v = bar["close"], bar["volume"]
        remaining, total_cost = qty, 0.0
        depth = self.cfg.depth_profile_usd / c

        levels = 5
        for i in range(levels):
            level_depth = depth / levels * self._rng.uniform(0.5, 1.5)
            level_price = c * (1 + (i + 1) * 0.0002 * (1 if side == "buy" else -1))
            filled = min(remaining, level_depth)
            total_cost += filled * level_price
            remaining -= filled
            if remaining <= 1e-12:
                break

        filled_qty = qty - remaining
        if filled_qty < 1e-12:
            return FillResult(0, 0, 0, 0, 0, "rejected")

        avg_price = total_cost / filled_qty
        fee = total_cost * self.cfg.taker_fee
        return FillResult(
            fill_price=avg_price, fill_qty=filled_qty, fee=fee,
            slippage_cost=abs(avg_price - c) * filled_qty,
            market_impact=self.estimate_market_impact(qty, c, v) * c * qty,
            status="filled" if remaining < 1e-12 else "partial",
        )

    @staticmethod
    def _position_pnl_pct(pos: Position, price: float) -> float:
        if pos.entry_price <= 0:
            return 0.0
        raw = (price - pos.entry_price) / pos.entry_price
        if pos.side == "short":
            raw = -raw
        return raw * pos.leverage
