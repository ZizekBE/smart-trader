"""ExecutionEngine — full trade lifecycle: signal → order → record → exit.

Responsibilities:
  • process_signal()  — place entry order + persist open trade
  • check_exits()     — evaluate SL/TP for all open positions
  • close_trade()     — place exit order + update trade record with P&L
  • get_portfolio()   — reconstruct Portfolio state from DB
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import structlog

from smart_trader.data.storage.database import get_session_factory
from smart_trader.data.storage.trade_repo import TradeRepository
from smart_trader.execution.orders.manager import OrderManager
from smart_trader.risk.models import OpenPosition, Portfolio, RiskDecision
from smart_trader.strategy.signals.models import SignalEvent

log = structlog.get_logger(__name__)

EXCHANGE = "gateio"
INITIAL_CASH = 10_000.0   # default paper portfolio value


@dataclass
class ClosedTrade:
    trade_id:    uuid.UUID
    symbol:      str
    side:        str
    pnl:         float
    pnl_pct:     float
    exit_reason: str
    exit_price:  float
    sleeve:      str = "tactical"


class ExecutionEngine:
    def __init__(self, client=None, paper: bool = True, exchange: str = EXCHANGE) -> None:
        self._order_mgr = OrderManager(client=client, paper=paper)
        self._factory   = get_session_factory()
        self._exchange  = exchange
        self._log       = log.bind(engine="execution", paper=paper)

    # ── entry ──────────────────────────────────────────────────

    async def process_signal(
        self,
        signal:       SignalEvent,
        decision:     RiskDecision,
        current_price: float,
        signal_db_id:  Optional[uuid.UUID] = None,
        sleeve:        str = "tactical",
    ) -> Optional[uuid.UUID]:
        """Place entry order and record the open trade.

        Returns the trade UUID on success, None on failure.
        """
        if not decision.approved or decision.position_size is None:
            self._log.debug("skipped_rejected", symbol=signal.symbol)
            return None

        pos   = decision.position_size
        side  = "buy" if signal.signal_type == "long" else "sell"

        result = await self._order_mgr.place(signal.symbol, side, pos.quantity, current_price)
        if result.status == "failed":
            self._log.error("fill_failed", symbol=signal.symbol, status=result.status)
            return None
        if result.status == "partial":
            self._log.warning(
                "partial_fill",
                symbol=signal.symbol,
                requested=pos.quantity, filled=result.quantity,
            )

        row = {
            "signal_id":   signal_db_id,
            "opened_at":   result.timestamp,
            "symbol":      signal.symbol,
            "exchange":    self._exchange,
            "side":        side,
            "entry_price": result.fill_price,
            "quantity":    result.quantity,
            "fees":        result.fee,
            "slippage":    result.slippage,
            "status":      "open",
            "sleeve":      sleeve,
            "metadata_col": {
                "stop_price":        pos.stop_price,
                "take_profit_price": pos.take_profit_price,
                "order_id":          result.order_id,
                "signal_confidence": signal.confidence,
                "regime":            str(signal.regime),
                "sleeve":            sleeve,
            },
        }

        async with self._factory() as session:
            repo     = TradeRepository(session)
            trade_id = await repo.open_trade(row)

        self._log.info(
            "trade_opened",
            trade_id=str(trade_id), symbol=signal.symbol, side=side,
            qty=result.quantity, fill=result.fill_price,
            sl=pos.stop_price, tp=pos.take_profit_price,
        )
        return trade_id

    # ── open trade queries ─────────────────────────────────────

    async def get_open_trades(self, sleeve: Optional[str] = None):
        """Return open Trade ORM objects, optionally filtered by sleeve."""
        async with self._factory() as session:
            repo = TradeRepository(session)
            return await repo.get_open(sleeve=sleeve)

    # ── exit monitoring ────────────────────────────────────────

    async def check_exits(
        self,
        current_prices: dict[str, float],
        sleeve: Optional[str] = None,
    ) -> list[ClosedTrade]:
        """Check all open trades for SL/TP triggers.

        Call this on every new candle / price tick.
        """
        async with self._factory() as session:
            repo        = TradeRepository(session)
            open_trades = await repo.get_open(sleeve=sleeve)

        closed: list[ClosedTrade] = []
        for trade in open_trades:
            price = current_prices.get(trade.symbol)
            if price is None:
                continue

            meta      = trade.metadata_col or {}
            stop_p    = meta.get("stop_price")
            tp_p      = meta.get("take_profit_price")
            exit_reason: Optional[str] = None

            if trade.side == "buy":   # long
                if stop_p and price <= stop_p:
                    exit_reason = "stop_loss"
                elif tp_p and price >= tp_p:
                    exit_reason = "take_profit"
            else:                     # short
                if stop_p and price >= stop_p:
                    exit_reason = "stop_loss"
                elif tp_p and price <= tp_p:
                    exit_reason = "take_profit"

            if exit_reason:
                ct = await self.close_trade(trade.id, price, exit_reason)
                if ct:
                    closed.append(ct)

        return closed

    async def close_trade(
        self,
        trade_id:    uuid.UUID,
        exit_price:  float,
        exit_reason: str,
    ) -> Optional[ClosedTrade]:
        """Place exit order and update the trade record with final P&L.

        Uses a single DB session with SELECT FOR UPDATE to prevent concurrent
        close_trade calls from double-closing the same trade.
        """
        async with self._factory() as session:
            repo  = TradeRepository(session)

            # Lock the row — any concurrent close_trade will block here
            trade = await repo.get_for_update(trade_id)
            if trade is None or trade.status != "open":
                self._log.debug("close_skipped_not_open", trade_id=str(trade_id))
                return None

            # Snapshot values before leaving the session scope
            symbol   = trade.symbol
            side     = trade.side
            qty      = float(trade.quantity)
            entry    = float(trade.entry_price)
            prev_fee = float(trade.fees)
            prev_slip = float(trade.slippage or 0)
            sleeve   = getattr(trade, "sleeve", "tactical")

            # Place exit order (opposite side)
            exit_side = "sell" if side == "buy" else "buy"
            result    = await self._order_mgr.place(symbol, exit_side, qty, exit_price)

            total_fees = prev_fee + result.fee
            if side == "buy":
                pnl = (result.fill_price - entry) * qty - total_fees
            else:
                pnl = (entry - result.fill_price) * qty - total_fees
            pnl_pct = pnl / (entry * qty) if entry * qty > 0 else 0.0

            # Close within the same locked transaction
            closed = await repo.close_trade(
                trade_id=trade_id,
                exit_price=result.fill_price,
                closed_at=result.timestamp,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=total_fees,
                slippage=prev_slip + result.slippage,
                exit_reason=exit_reason,
            )

        if not closed:
            self._log.warning("close_race_detected", trade_id=str(trade_id))
            return None

        self._log.info(
            "trade_closed",
            trade_id=str(trade_id), symbol=symbol,
            reason=exit_reason, pnl=round(pnl, 4), pnl_pct=f"{pnl_pct:.2%}",
        )
        return ClosedTrade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            exit_reason=exit_reason,
            exit_price=result.fill_price,
            sleeve=sleeve,
        )

    # ── portfolio state ────────────────────────────────────────

    async def get_portfolio(
        self,
        initial_cash:   float = INITIAL_CASH,
        current_prices: Optional[dict[str, float]] = None,
        sleeve:         Optional[str] = None,
        sleeve_budget:  Optional[float] = None,
        peak_key:       str = "default",
    ) -> Portfolio:
        """Reconstruct Portfolio from open DB trades.

        When ``sleeve`` is given, ``open_positions`` is filtered to that sleeve
        only (for duplicate-position checks), but ``total_value`` and
        ``daily_pnl`` always reflect the combined portfolio.

        ``sleeve_budget`` caps the available cash for this sleeve so the
        PositionSizer respects the sleeve's capital allocation.

        ``peak_value`` is persisted in the ``portfolio_state`` table so that
        drawdown calculation survives process restarts.
        """
        current_prices = current_prices or {}

        async with self._factory() as session:
            repo            = TradeRepository(session)
            all_open        = await repo.get_open()
            sleeve_open     = await repo.get_open(sleeve=sleeve) if sleeve else all_open
            closed          = await repo.get_closed(limit=500)
            stored_peak     = await repo.get_peak_value(key=peak_key)

        today = date.today()
        daily_pnl = sum(
            float(t.pnl or 0) for t in closed
            if t.closed_at and t.closed_at.date() == today
        )

        all_positions_unrealised = 0.0
        all_used = 0.0
        for t in all_open:
            price = current_prices.get(t.symbol, float(t.entry_price))
            qty   = float(t.quantity)
            ep    = float(t.entry_price)
            side  = "long" if t.side == "buy" else "short"
            unr   = qty * (price - ep) if side == "long" else qty * (ep - price)
            all_positions_unrealised += unr
            all_used += qty * price

        total_value = (initial_cash
                       + sum(float(t.pnl or 0) for t in closed)
                       + all_positions_unrealised)

        # Resolve peak: max(initial, stored, current)
        peak_value = max(initial_cash, stored_peak or 0.0, total_value)

        # Persist new peak if it increased
        if peak_value > (stored_peak or 0.0):
            async with self._factory() as session:
                repo = TradeRepository(session)
                await repo.update_peak_value(peak_value, key=peak_key)

        # Build positions dict filtered to this sleeve
        positions: dict[str, OpenPosition] = {}
        sleeve_used = 0.0
        for t in sleeve_open:
            price = current_prices.get(t.symbol, float(t.entry_price))
            pos   = OpenPosition(
                symbol=t.symbol,
                side="long" if t.side == "buy" else "short",
                quantity=float(t.quantity),
                entry_price=float(t.entry_price),
                current_price=price,
            )
            key = f"{t.symbol}:{getattr(t, 'sleeve', 'tactical')}"
            positions[key] = pos
            sleeve_used += pos.notional

        if sleeve_budget is not None:
            cash = max(0.0, sleeve_budget - sleeve_used)
        else:
            cash = max(0.0, total_value - all_used)

        return Portfolio(
            total_value=total_value,
            cash=cash,
            open_positions=positions,
            peak_value=peak_value,
            daily_pnl=daily_pnl,
            daily_start_value=total_value - daily_pnl,
        )
