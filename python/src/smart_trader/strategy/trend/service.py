"""TrendService — fetches candles from DB and runs the TrendEngine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog

from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import TrendState

log = structlog.get_logger(__name__)

# How many bars to load per timeframe for a reliable analysis
_LOOKBACK: dict[str, int] = {
    "1m":  500,
    "5m":  500,
    "15m": 300,
    "1h":  300,
    "4h":  200,
    "1d":  200,
    "1w":  100,
}
_DEFAULT_LOOKBACK = 300


class TrendService:
    """Combines CandleRepository + TrendEngine.

    Fetches the required candle window from the DB, converts to a
    DataFrame, and returns a TrendState for the requested symbol/timeframe.
    """

    def __init__(self, exchange: str = "gateio") -> None:
        self._factory = get_session_factory()
        self._engine  = TrendEngine()
        self._exchange = exchange
        self._log = log.bind(service="trend")

    async def analyse(
        self,
        symbol: str,
        timeframe: str,
    ) -> TrendState:
        lookback = _LOOKBACK.get(timeframe, _DEFAULT_LOOKBACK)
        now   = datetime.now(timezone.utc)
        since = now - _timeframe_delta(timeframe, lookback)

        async with self._factory() as session:
            repo = CandleRepository(session)
            candles = await repo.get_range(symbol, self._exchange, timeframe, since, now)

        if not candles:
            raise RuntimeError(
                f"No candles in DB for {symbol}/{timeframe} — run backfill first"
            )

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

        self._log.debug(
            "analyse_start",
            symbol=symbol, tf=timeframe, bars=len(df),
        )

        state = self._engine.analyse(df, symbol, timeframe)

        self._log.info(
            "trend_state",
            symbol=symbol, tf=timeframe,
            regime=state.regime, mode=state.strategy_mode,
            direction=state.direction, strength=state.strength,
        )
        return state

    async def analyse_multi(
        self,
        symbol: str,
        timeframes: list[str],
    ) -> dict[str, TrendState]:
        """Analyse multiple timeframes for the same symbol."""
        results: dict[str, TrendState] = {}
        for tf in timeframes:
            try:
                results[tf] = await self.analyse(symbol, tf)
            except Exception as exc:
                self._log.error("analyse_failed", symbol=symbol, tf=tf, error=str(exc))
        return results


def _timeframe_delta(timeframe: str, bars: int) -> timedelta:
    """Convert (timeframe, bar_count) to a timedelta."""
    minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
    }
    return timedelta(minutes=minutes.get(timeframe, 60) * bars)
