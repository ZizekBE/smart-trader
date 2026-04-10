"""SignalService — full pipeline: candles → trend → signals → DB.

Typical usage::
    service = SignalService()
    events  = await service.run(symbol="BTC/USDT", signal_tf="1h", trend_tf="1d")
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import structlog

from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.data.storage.signal_repo import SignalRepository
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.signals.models import SignalEvent
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import TrendState
from smart_trader.strategy.trend.service import TrendService, _timeframe_delta

log = structlog.get_logger(__name__)

# bars to load for signal detection (short window is enough)
_SIGNAL_LOOKBACK: dict[str, int] = {
    "1m":  100,
    "5m":  100,
    "15m": 80,
    "1h":  80,
    "4h":  60,
}
_DEFAULT_SIGNAL_LOOKBACK = 80


class SignalService:
    """Orchestrates trend analysis + signal detection for one symbol."""

    def __init__(self, exchange: str = "gateio") -> None:
        self._factory       = get_session_factory()
        self._trend_service = TrendService(exchange=exchange)
        self._signal_engine = SignalEngine()
        self._exchange      = exchange
        self._log           = log.bind(service="signal")

    async def run(
        self,
        symbol: str,
        signal_tf: str = "1h",
        trend_tf:  str = "1d",
        persist:   bool = True,
        min_confidence: float = 0.40,
    ) -> list[SignalEvent]:
        """Run the full signal pipeline for one symbol.

        Args:
            symbol:         e.g. 'BTC/USDT'
            signal_tf:      timeframe for signal detection (1m, 1h…)
            trend_tf:       timeframe for trend context (1d, 1w…)
            persist:        if True, save signals to the DB
            min_confidence: drop signals below this threshold

        Returns:
            List of SignalEvents sorted by confidence descending.
        """
        # ── 1. trend context (higher timeframe) ───────────────
        try:
            trend_state: TrendState = await self._trend_service.analyse(symbol, trend_tf)
        except Exception as exc:
            self._log.error("trend_failed", symbol=symbol, tf=trend_tf, error=str(exc))
            return []

        # ── 2. signal-timeframe candles from DB ───────────────
        lookback = _SIGNAL_LOOKBACK.get(signal_tf, _DEFAULT_SIGNAL_LOOKBACK)
        now      = datetime.now(timezone.utc)
        since    = now - _timeframe_delta(signal_tf, lookback)

        async with self._factory() as session:
            repo    = CandleRepository(session)
            candles = await repo.get_range(symbol, self._exchange, signal_tf, since, now)

        if not candles:
            self._log.warning("no_candles", symbol=symbol, tf=signal_tf)
            return []

        df = pd.DataFrame(
            [
                {
                    "time":   c.time,
                    "open":   float(c.open),
                    "high":   float(c.high),
                    "low":    float(c.low),
                    "close":  float(c.close),
                    "volume": float(c.volume),
                }
                for c in candles
            ]
        ).sort_values("time").reset_index(drop=True)

        # ── 3. detect signals ─────────────────────────────────
        events = self._signal_engine.analyse(
            df, symbol, signal_tf, trend_state,
            min_confidence=min_confidence,
        )

        self._log.info(
            "signals_detected",
            symbol=symbol, signal_tf=signal_tf, trend_tf=trend_tf,
            regime=trend_state.regime, count=len(events),
        )

        # ── 4. persist to DB ──────────────────────────────────
        if persist and events:
            rows = [e.to_db_row() for e in events]
            async with self._factory() as session:
                repo = SignalRepository(session)
                await repo.insert_many(rows)
            self._log.info("signals_saved", count=len(rows))

        return events

    async def run_multi(
        self,
        symbols: list[str],
        signal_tf: str = "1h",
        trend_tf:  str = "1d",
        persist:   bool = True,
    ) -> dict[str, list[SignalEvent]]:
        """Run signal detection for multiple symbols."""
        results: dict[str, list[SignalEvent]] = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.run(symbol, signal_tf, trend_tf, persist)
            except Exception as exc:
                self._log.error("run_failed", symbol=symbol, error=str(exc))
                results[symbol] = []
        return results
