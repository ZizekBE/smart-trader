"""PerformanceAnalyzer — DB-backed façade for analysis and backtesting.

Two modes
─────────
1. Live analysis  — reads closed trades from the database, computes metrics.
2. Backtest       — replays pipeline on historical candles in memory.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from smart_trader.analysis.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from smart_trader.analysis.metrics.calculator import MetricsCalculator, TradeResult
from smart_trader.analysis.reporting.report import PerformanceReport
from smart_trader.data.storage.database import get_session_factory
from smart_trader.data.storage.trade_repo import TradeRepository
from smart_trader.data.storage.candle_repo import CandleRepository


class PerformanceAnalyzer:
    """High-level analysis façade.

    Usage::
        analyzer = PerformanceAnalyzer()

        # analyse closed DB trades
        report = await analyzer.analyse_live("BTC/USDT", initial_capital=10_000)
        print(report.ascii())

        # run backtest on historical candles
        result = await analyzer.backtest("BTC/USDT", "1h", initial_capital=10_000)
        print(result.metrics)
    """

    def __init__(self) -> None:
        self._factory = get_session_factory()
        self._calc    = MetricsCalculator()

    # ── live analysis ─────────────────────────────────────────────────────────

    async def analyse_live(
        self,
        symbol:          Optional[str]   = None,
        initial_capital: float           = 10_000.0,
        since:           Optional[datetime] = None,
        until:           Optional[datetime] = None,
        limit:           int             = 5_000,
    ) -> PerformanceReport:
        """Read closed trades from DB and return a PerformanceReport."""
        async with self._factory() as session:
            repo   = TradeRepository(session)
            trades = await repo.get_closed(symbol=symbol, since=since, until=until, limit=limit)

        trade_results = [
            TradeResult(
                pnl=float(t.pnl or 0),
                pnl_pct=float(t.pnl_pct or 0),
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                fees=float(t.fees or 0),
                slippage=float(t.slippage or 0),
                side=t.side,
            )
            for t in trades
            if t.closed_at is not None
        ]

        metrics = self._calc.calculate(trade_results, initial_capital)

        start_at = min((r.opened_at for r in trade_results), default=None)
        end_at   = max((r.closed_at for r in trade_results), default=None)

        return PerformanceReport(
            metrics=metrics,
            title="Live Performance Report",
            start_at=start_at,
            end_at=end_at,
        )

    # ── backtest ──────────────────────────────────────────────────────────────

    async def backtest(
        self,
        symbol:           str,
        timeframe:        str   = "1h",
        trend_timeframe:  str   = "1d",
        mid_timeframe:    str   = "4h",
        initial_capital:  float = 10_000.0,
        since:            Optional[datetime] = None,
        until:            Optional[datetime] = None,
        exchange:         str   = "gateio",
        min_confidence:   float = 0.65,
        max_position_pct: float = 0.05,
        sl_cooldown_bars: int   = 3,
        require_mtf:      bool  = True,
        strategy_version: str   = "v2",
        # risk param overrides from Bayesian optimisation
        atr_mult:              Optional[float] = None,
        rr_ratio:              Optional[float] = None,
        kelly_frac:            Optional[float] = None,
        trailing_atr_mult:     Optional[float] = None,
        trailing_trigger_pct:  Optional[float] = None,
        breakeven_trigger_pct: Optional[float] = None,
        pullback_atr_frac:     Optional[float] = None,
        min_confidence_high_vol: Optional[float] = None,
    ) -> BacktestResult:
        """Fetch candles from DB and run a full in-memory backtest."""
        from datetime import timezone
        _since = since or datetime(2000, 1, 1, tzinfo=timezone.utc)
        _until = until or datetime(2100, 1, 1, tzinfo=timezone.utc)

        async with self._factory() as session:
            repo = CandleRepository(session)
            sig_candles   = await repo.get_range(
                symbol, exchange, timeframe, _since, _until
            )
            trend_candles = await repo.get_range(
                symbol, exchange, trend_timeframe, _since, _until
            )
            mid_candles   = await repo.get_range(
                symbol, exchange, mid_timeframe, _since, _until
            )

        sig_df   = self._candles_to_df(sig_candles)
        trend_df = self._candles_to_df(trend_candles) if trend_candles else None
        mid_df   = self._candles_to_df(mid_candles)   if mid_candles   else None

        cfg = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            trend_timeframe=trend_timeframe,
            initial_capital=initial_capital,
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
            sl_cooldown_bars=sl_cooldown_bars,
            require_mtf=require_mtf,
            strategy_version=strategy_version,
            atr_mult=atr_mult,
            rr_ratio=rr_ratio,
            kelly_frac=kelly_frac,
            trailing_atr_mult=trailing_atr_mult,
            trailing_trigger_pct=trailing_trigger_pct,
            breakeven_trigger_pct=breakeven_trigger_pct,
            pullback_atr_frac=pullback_atr_frac,
            min_confidence_high_vol=min_confidence_high_vol if min_confidence_high_vol is not None else 0.75,
        )
        engine = BacktestEngine(cfg)
        return engine.run(sig_df, trend_df, mid_df)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        rows = [
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
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("time")
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
        return df
