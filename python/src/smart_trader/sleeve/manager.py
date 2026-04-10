"""SleeveManager — coordinates CoreSleeve and TacticalSleeve.

Responsibilities:
  • Build per-sleeve Portfolio views (scoped cash + combined total_value)
  • Check ConflictGuard before approving entries
  • Route approved entries to ExecutionEngine with correct sleeve tag
  • Delegate exit checks (SL/TP/time-stop) and update cooldown state
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd
import structlog

from smart_trader.execution.engine import ExecutionEngine
from smart_trader.sleeve.capital import CapitalAllocator
from smart_trader.sleeve.conflict import ConflictGuard
from smart_trader.sleeve.long_sleeve import CoreSleeve
from smart_trader.sleeve.models import SleeveId
from smart_trader.sleeve.short_sleeve import TacticalSleeve

log = structlog.get_logger(__name__)


class SleeveManager:
    """Orchestrates both sleeves for a single symbol.

    The manager owns both sleeve instances and routes their decisions through
    capital allocation and conflict checking before execution.
    """

    def __init__(
        self,
        symbol:        str,
        initial_cash:  float,
        engine:        ExecutionEngine,
        core:          CoreSleeve,
        tactical:      TacticalSleeve,
        allocator:     CapitalAllocator,
        signal_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._symbol   = symbol
        self._engine   = engine
        self._core     = core
        self._tactical = tactical
        self._alloc    = allocator
        self._conflict = ConflictGuard()
        self._initial  = initial_cash
        self._signal_callback = signal_callback
        self._log = log.bind(symbol=symbol, component="sleeve_manager")

    # ── public entry point ─────────────────────────────────────────────────────

    async def tick_exits(self, price: float) -> None:
        """Process time-stops and SL/TP exits. Call before sleeve evaluations."""
        prices = {self._symbol: price}

        await self._check_time_stops(price)

        closed = await self._engine.check_exits(prices)
        for ct in closed:
            self._log.info(
                "trade_closed",
                sleeve=ct.sleeve, reason=ct.exit_reason,
                pnl=ct.pnl, trade_id=str(ct.trade_id)[:8],
            )
            side = "buy" if ct.side == "buy" else "sell"
            if ct.exit_reason == "stop_loss":
                if ct.sleeve == SleeveId.CORE:
                    self._core.on_stop_loss(side)
                else:
                    self._tactical.on_stop_loss(side)

        self._core.tick_down_cooldowns()
        self._tactical.tick_down_cooldowns()

    async def tick_tactical(
        self,
        price:    float,
        sig_df:   pd.DataFrame,
        trend_df: pd.DataFrame,
        mid_df:   pd.DataFrame,
    ) -> None:
        """Evaluate and potentially execute the tactical sleeve."""
        prices = {self._symbol: price}
        core_portfolio = await self._engine.get_portfolio(
            initial_cash=self._initial, current_prices=prices,
            sleeve=SleeveId.CORE, sleeve_budget=self._alloc.long_budget,
        )
        tactical_portfolio = await self._engine.get_portfolio(
            initial_cash=self._initial, current_prices=prices,
            sleeve=SleeveId.TACTICAL, sleeve_budget=self._alloc.short_budget,
        )

        decision = self._tactical.evaluate(sig_df, trend_df, mid_df, tactical_portfolio, price)
        self._emit(decision.tick_meta)

        if decision.signal and decision.decision and decision.decision.approved:
            pos = decision.decision.position_size
            allowed, reason = self._conflict.check(
                new_sleeve="tactical",
                new_side="buy" if decision.signal.signal_type == "long" else "sell",
                new_notional=pos.notional if pos else 0,
                core_portfolio=core_portfolio,
                tactical_portfolio=tactical_portfolio,
            )
            if allowed:
                trade_id = await self._engine.process_signal(
                    decision.signal, decision.decision, price, sleeve=SleeveId.TACTICAL,
                )
                if trade_id:
                    self._log.info(
                        "tactical_trade_opened", trade_id=str(trade_id)[:8],
                        side=decision.tick_meta.get("entry_side"),
                        confidence=decision.tick_meta.get("entry_confidence"),
                    )
            else:
                self._log.info("tactical_conflict_blocked", reason=reason)

    async def tick_core(
        self,
        price:    float,
        sig_df:   pd.DataFrame,   # 4h data for core sleeve
        trend_df: pd.DataFrame,
    ) -> None:
        """Evaluate and potentially execute the core sleeve (4h boundary)."""
        prices = {self._symbol: price}
        core_portfolio = await self._engine.get_portfolio(
            initial_cash=self._initial, current_prices=prices,
            sleeve=SleeveId.CORE, sleeve_budget=self._alloc.long_budget,
        )
        tactical_portfolio = await self._engine.get_portfolio(
            initial_cash=self._initial, current_prices=prices,
            sleeve=SleeveId.TACTICAL, sleeve_budget=self._alloc.short_budget,
        )

        decision = self._core.evaluate(sig_df, trend_df, core_portfolio, price)
        self._emit(decision.tick_meta)

        if decision.signal and decision.decision and decision.decision.approved:
            pos = decision.decision.position_size
            allowed, reason = self._conflict.check(
                new_sleeve="core",
                new_side="buy" if decision.signal.signal_type == "long" else "sell",
                new_notional=pos.notional if pos else 0,
                core_portfolio=core_portfolio,
                tactical_portfolio=tactical_portfolio,
            )
            if allowed:
                trade_id = await self._engine.process_signal(
                    decision.signal, decision.decision, price, sleeve=SleeveId.CORE,
                )
                if trade_id:
                    self._log.info(
                        "core_trade_opened", trade_id=str(trade_id)[:8],
                        side=decision.tick_meta.get("entry_side"),
                        confidence=decision.tick_meta.get("entry_confidence"),
                    )
            else:
                self._log.info("core_conflict_blocked", reason=reason)

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _check_time_stops(self, price: float) -> None:
        """Force-close tactical trades that have exceeded their max hold duration."""
        trades = await self._engine.get_open_trades(sleeve=SleeveId.TACTICAL)

        now = datetime.now(timezone.utc)
        for trade in trades:
            opened = trade.opened_at
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if self._tactical.should_time_stop(opened):
                self._log.info(
                    "time_stop",
                    trade_id=str(trade.id)[:8],
                    symbol=trade.symbol,
                    held_h=round((now - opened).total_seconds() / 3600, 1),
                )
                await self._engine.close_trade(trade.id, price, "time_stop")

    def _emit(self, meta: dict) -> None:
        if self._signal_callback:
            record = dict(meta)
            record.setdefault("time", datetime.now(timezone.utc).isoformat())
            record.setdefault("symbol", self._symbol)
            self._signal_callback(record)
