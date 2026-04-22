"""MultiHybridLoop — runs HybridLoop for multiple symbols concurrently.

Each symbol gets its own independent sleeve pair (CoreSleeve + TacticalSleeve)
sharing a single GateIO client and DB pool. Capital is split evenly across symbols.

Correlation guard: if the rolling 30-day ETH/BTC correlation exceeds 0.80,
both loops apply a 20% position-cap reduction for that tick.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import structlog

from smart_trader.core.settings import get_settings
from smart_trader.data.ingestion.gateio_client import GateIOClient
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.sleeve.correlation_guard import CorrelationGuard
from smart_trader.trader.hybrid_loop import HybridLoop

log = structlog.get_logger(__name__)

_TREND_TF = "1d"


class MultiHybridLoop:
    """Runs one HybridLoop per symbol, with cross-symbol correlation guard.

    Usage::
        loop = MultiHybridLoop(symbols=["ETH/USDT", "BTC/USDT"], total_capital=10_000)
        await loop.run()
    """

    def __init__(
        self,
        symbols:       list[str],
        total_capital: float = 10_000.0,
        paper:         bool  = True,
    ) -> None:
        self._symbols       = symbols
        self._total_capital = total_capital
        self._paper         = paper
        self._factory       = get_session_factory()
        self._corr_guard    = CorrelationGuard()
        self._log           = log.bind(mode="multi", symbols=symbols, paper=paper)

        capital_per_symbol = total_capital / len(symbols)
        self._loops: dict[str, HybridLoop] = {
            sym: HybridLoop(symbol=sym, initial_cash=capital_per_symbol, paper=paper)
            for sym in symbols
        }

    async def run(self) -> None:
        self._log.info(
            "multi_loop_started",
            symbols=self._symbols,
            capital_per_symbol=self._total_capital / len(self._symbols),
        )
        # Apply initial correlation check, then run all loops concurrently.
        # Each loop runs its own sleep/wake cycle aligned to its signal TF.
        await asyncio.gather(
            *[loop.run() for loop in self._loops.values()],
            self._correlation_watchdog(),
        )

    async def run_once(self) -> None:
        """Single-tick for testing."""
        await asyncio.gather(*[loop.run_once() for loop in self._loops.values()])

    async def _correlation_watchdog(self) -> None:
        """Runs hourly, updates correlation-based cap multiplier on all loops."""
        while True:
            try:
                await self._update_corr_caps()
            except Exception as exc:
                self._log.warning("corr_watchdog_error", error=str(exc))
            await asyncio.sleep(3600)

    async def _update_corr_caps(self) -> None:
        if len(self._symbols) < 2:
            return

        dfs: dict[str, pd.DataFrame] = {}
        now   = datetime.now(timezone.utc)
        since = now - timedelta(days=45)

        async with self._factory() as session:
            repo = CandleRepository(session)
            for sym in self._symbols:
                candles = await repo.get_range(sym, "gateio", _TREND_TF, since, now)
                if candles:
                    rows = [{"time": c.time, "close": float(c.close)} for c in candles]
                    df   = pd.DataFrame(rows).set_index("time")
                    df.index = pd.to_datetime(df.index, utc=True)
                    dfs[sym] = df

        if len(dfs) < 2:
            return

        sym_a, sym_b = self._symbols[0], self._symbols[1]
        mult = self._corr_guard.position_cap_mult(
            dfs.get(sym_a, pd.DataFrame()),
            dfs.get(sym_b, pd.DataFrame()),
        )

        # Propagate multiplier to each loop's regime adapter by injecting it
        # as an override on the CapitalAllocator's budgets.
        for sym, loop in self._loops.items():
            alloc = loop._manager._alloc
            original_long  = loop._initial_cash * get_settings().hybrid_long_budget_pct
            original_short = loop._initial_cash * get_settings().hybrid_short_budget_pct
            alloc._long_budget  = original_long  * mult
            alloc._short_budget = original_short * mult

        if mult < 1.0:
            self._log.info(
                "correlation_cap_applied",
                symbols=self._symbols,
                mult=round(mult, 2),
            )
