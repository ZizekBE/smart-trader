"""OrderManager — paper-mode and live order placement.

Paper mode:
  Fills immediately at current price with simulated slippage + fee.
  No exchange connection needed.

Live mode:
  Delegates to GateIOClient.  Handles:
    • Order status polling (up to ORDER_POLL_TIMEOUT_S)
    • Partial fills — cancels remainder, reports filled quantity
    • Rejected / expired orders — returns status="failed"
    • Real exchange fee extraction from the CCXT response
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Gate.io spot taker fee + typical slippage (used for paper mode only)
FEE_RATE      = 0.001    # 0.10 %
SLIPPAGE_RATE = 0.0005   # 0.05 % per side

ORDER_POLL_INTERVAL_S = 1.0
ORDER_POLL_TIMEOUT_S  = 30.0

# CCXT statuses that mean the order is no longer open
_TERMINAL_STATUSES = frozenset({"closed", "canceled", "cancelled", "expired", "rejected"})


@dataclass(slots=True)
class OrderResult:
    order_id:    str
    symbol:      str
    side:        str      # 'buy' | 'sell'
    quantity:    float    # actually filled quantity (may be < requested)
    fill_price:  float
    fee:         float    # absolute fee in quote currency
    slippage:    float    # absolute slippage cost in quote currency
    status:      str      # 'filled' | 'partial' | 'failed'
    timestamp:   datetime


class OrderManager:
    """Place and manage orders.

    Inject a GateIOClient for live mode; leave it None for paper mode.
    """

    def __init__(self, client=None, paper: bool = True) -> None:
        self._client = client
        self._paper  = paper
        self._log    = log.bind(component="order_manager", paper=paper)

    async def place(
        self,
        symbol:   str,
        side:     str,     # 'buy' | 'sell'
        quantity: float,
        price:    float,
    ) -> OrderResult:
        if self._paper:
            return self._paper_fill(symbol, side, quantity, price)
        return await self._live_fill(symbol, side, quantity, price)

    async def cancel(self, order_id: str, symbol: str) -> bool:
        if self._paper:
            self._log.info("paper_cancel", order_id=order_id)
            return True
        if self._client is None:
            return False
        await self._client.cancel_order(order_id, symbol)
        return True

    # ── live order handling ────────────────────────────────────

    async def _live_fill(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float,
    ) -> OrderResult:
        if self._client is None:
            raise RuntimeError("GateIOClient required for live trading")

        raw = await self._client.place_order(symbol, side, quantity, price)
        order_id = str(raw.get("id", ""))

        if not order_id:
            self._log.error("order_no_id", symbol=symbol, side=side, raw=raw)
            return self._failed_result(symbol, side, quantity, price)

        # Poll until the order reaches a terminal state or times out
        order = await self._poll_order(order_id, symbol)

        status_str = str(order.get("status", "")).lower()
        filled_qty = float(order.get("filled", 0) or 0)
        avg_price  = float(order.get("average", 0) or order.get("price", 0) or price)

        # Extract real fee from CCXT response
        fee_info = order.get("fee") or {}
        fee_cost = float(fee_info.get("cost", 0) or 0)
        if fee_cost == 0 and filled_qty > 0:
            fee_cost = avg_price * filled_qty * FEE_RATE

        slippage_abs = abs(avg_price - price) * filled_qty if filled_qty > 0 else 0.0

        if filled_qty <= 0:
            self._log.warning(
                "order_not_filled",
                order_id=order_id, status=status_str,
                symbol=symbol, side=side,
            )
            return self._failed_result(symbol, side, quantity, price, order_id=order_id)

        # Cancel unfilled remainder for partial fills
        if filled_qty < quantity * 0.999 and status_str not in _TERMINAL_STATUSES:
            try:
                await self._client.cancel_order(order_id, symbol)
                self._log.info(
                    "partial_fill_remainder_cancelled",
                    order_id=order_id,
                    filled=filled_qty, requested=quantity,
                )
            except Exception as exc:
                self._log.warning("cancel_remainder_failed", error=str(exc))

        result_status = "filled" if filled_qty >= quantity * 0.999 else "partial"

        self._log.info(
            "live_fill",
            order_id=order_id, symbol=symbol, side=side,
            requested=quantity, filled=filled_qty,
            avg_price=avg_price, fee=fee_cost, status=result_status,
        )
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=round(filled_qty, 8),
            fill_price=round(avg_price, 8),
            fee=round(fee_cost, 8),
            slippage=round(slippage_abs, 8),
            status=result_status,
            timestamp=datetime.now(timezone.utc),
        )

    async def _poll_order(self, order_id: str, symbol: str) -> dict:
        """Poll order status until terminal or timeout."""
        elapsed = 0.0
        order: dict = {}
        while elapsed < ORDER_POLL_TIMEOUT_S:
            try:
                order = await self._client.get_order(order_id, symbol)
            except Exception as exc:
                self._log.warning("poll_failed", order_id=order_id, error=str(exc))
                await asyncio.sleep(ORDER_POLL_INTERVAL_S)
                elapsed += ORDER_POLL_INTERVAL_S
                continue

            status = str(order.get("status", "")).lower()
            if status in _TERMINAL_STATUSES:
                return order

            await asyncio.sleep(ORDER_POLL_INTERVAL_S)
            elapsed += ORDER_POLL_INTERVAL_S

        self._log.warning("poll_timeout", order_id=order_id, elapsed=elapsed)
        return order

    @staticmethod
    def _failed_result(
        symbol: str, side: str, quantity: float, price: float,
        order_id: str = "",
    ) -> OrderResult:
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=0.0,
            fill_price=price,
            fee=0.0,
            slippage=0.0,
            status="failed",
            timestamp=datetime.now(timezone.utc),
        )

    # ── paper-mode helpers ────────────────────────────────────

    def _paper_fill(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Simulate a market fill with slippage."""
        slip_abs   = price * SLIPPAGE_RATE
        fill_price = (price + slip_abs) if side == "buy" else (price - slip_abs)
        fee        = fill_price * quantity * FEE_RATE
        slippage   = abs(fill_price - price) * quantity
        order_id   = f"paper_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        self._log.info(
            "paper_fill",
            symbol=symbol, side=side, qty=quantity,
            price=price, fill=fill_price, fee=fee,
        )
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=round(fill_price, 8),
            fee=round(fee, 8),
            slippage=round(slippage, 8),
            status="filled",
            timestamp=datetime.now(timezone.utc),
        )
