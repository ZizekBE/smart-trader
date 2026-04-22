"""HybridLoop — dual-sleeve trading loop.

Wakes every SHORT_TF candle close (default 1h).
  • Every tick    → TacticalSleeve evaluation
  • Every 4 ticks → CoreSleeve evaluation (4h boundary)

Both sleeves share a single GateIO client and DB connection pool.
Capital is partitioned via CapitalAllocator (60% core / 30% tactical / 10% reserve).
"""
from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import structlog

from smart_trader.core.settings import get_settings
from smart_trader.data.ingestion.candle_service import CandleIngestionService
from smart_trader.data.ingestion.gateio_client import GateIOClient
from smart_trader.data.storage.benchmark_repo import BenchmarkRepository
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.execution.engine import ExecutionEngine
from smart_trader.sleeve.capital import CapitalAllocator
from smart_trader.sleeve.long_sleeve import CoreSleeve
from smart_trader.sleeve.manager import SleeveManager
from smart_trader.sleeve.short_sleeve import TacticalSleeve
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import MarketRegime

log = structlog.get_logger(__name__)

SYMBOL        = "BTC/USDT"
SHORT_TF      = "1h"     # tactical + wake interval
LONG_TF       = "4h"     # core sleeve signal TF
TREND_TF      = "1d"     # shared trend filter
INITIAL_CASH  = 10_000.0
MIN_CANDLES   = 60
EXIT_CHECK_INTERVAL = 300   # seconds between mid-cycle exit checks

# How many SHORT_TF ticks before running the core sleeve
# 1h × 4 = 4h boundary
_CORE_EVERY_N_TICKS = 4

_NOTIFY_REGIMES = {MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING, MarketRegime.ACCUMULATION}


class HybridLoop:
    """Paper/live hybrid dual-sleeve trading loop.

    Usage::
        loop = HybridLoop(paper=True)
        await loop.run()
    """

    def __init__(
        self,
        symbol:          str                                       = SYMBOL,
        signal_tf:       str                                       = SHORT_TF,  # tactical TF
        initial_cash:    float                                     = INITIAL_CASH,
        paper:           bool                                      = True,
        signal_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._symbol       = symbol
        self._signal_tf    = signal_tf    # short sleeve TF = wake interval
        self._initial_cash = initial_cash
        self._paper        = paper
        self._signal_callback = signal_callback

        self._log = log.bind(symbol=symbol, mode="hybrid", paper=paper)

        s = get_settings()
        api_key    = s.gateio_api_key.get_secret_value()
        api_secret = s.gateio_api_secret.get_secret_value()

        self._client  = GateIOClient(api_key=api_key, api_secret=api_secret, paper=paper)
        self._svc     = CandleIngestionService(self._client)
        self._factory = get_session_factory()
        self._engine  = ExecutionEngine(client=self._client, paper=paper)

        # capital allocation
        allocator = CapitalAllocator(
            initial_capital=initial_cash,
            long_budget_pct=s.hybrid_long_budget_pct,
            short_budget_pct=s.hybrid_short_budget_pct,
        )

        # core sleeve
        core = CoreSleeve(
            symbol=symbol,
            signal_tf=s.hybrid_long_signal_tf,
            trend_tf=s.hybrid_long_trend_tf,
            min_confidence=s.hybrid_long_min_conf,
            max_position_pct=s.hybrid_long_max_pos_pct,
            atr_mult=s.hybrid_long_atr_mult,
            rr_ratio=s.hybrid_long_rr_ratio,
            strategy_version=s.strategy_version,
        )

        # tactical sleeve
        tactical = TacticalSleeve(
            symbol=symbol,
            signal_tf=s.hybrid_short_signal_tf,
            trend_tf=s.hybrid_short_trend_tf,
            mid_tf=s.hybrid_short_mid_tf,
            min_confidence=s.hybrid_short_min_conf,
            max_position_pct=s.hybrid_short_max_pos_pct,
            time_stop_hours=s.hybrid_short_time_stop_h,
            strategy_version=s.strategy_version,
        )

        self._manager = SleeveManager(
            symbol=symbol,
            initial_cash=initial_cash,
            engine=self._engine,
            core=core,
            tactical=tactical,
            allocator=allocator,
            signal_callback=signal_callback,
        )

        self._tick_count: int       = 0
        self._last_day: Optional[date] = None
        self._last_regime: Optional[MarketRegime] = None
        self._trend_eng = TrendEngine()
        self._benchmark_day: Optional[date] = None

    # ── public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Block forever, waking on each SHORT_TF candle close."""
        self._log.info(
            "hybrid_loop_started",
            signal_tf=self._signal_tf,
            core_every_n=_CORE_EVERY_N_TICKS,
        )
        try:
            while True:
                tick_start = datetime.now(timezone.utc)
                try:
                    await self._tick()
                except Exception as exc:
                    self._log.error("tick_error", error=str(exc), exc_info=True)
                await self._sleep_until_next_candle(tick_start)
        except asyncio.CancelledError:
            pass
        finally:
            await self._client.close()
            self._log.info("hybrid_loop_stopped")

    async def run_once(self) -> None:
        try:
            await self._tick()
        finally:
            await self._client.close()

    # ── private ───────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        self._tick_count += 1
        now = datetime.now(timezone.utc)
        run_core = (self._tick_count % _CORE_EVERY_N_TICKS == 0)

        self._log.info(
            "tick_begin",
            tick=self._tick_count,
            run_core=run_core,
            time=now.isoformat(),
        )

        # ── daily maintenance ──────────────────────────────────────────────
        today = now.date()
        if self._last_day != today:
            # Reset circuit breakers in both sleeve RiskManagers
            self._manager._core._risk.reset_breaker()
            self._manager._tactical._risk.reset_breaker()
            self._log.info("circuit_breakers_reset", date=str(today))
            self._last_day = today

        # ── sync candles ───────────────────────────────────────────────────
        timeframes_to_sync = {
            self._signal_tf,            # tactical signal TF (1h)
            get_settings().hybrid_long_signal_tf,  # core signal TF (4h)
            TREND_TF,                   # shared trend (1d)
            get_settings().hybrid_short_mid_tf,    # mid TF (4h)
        }
        for tf in timeframes_to_sync:
            try:
                n = await self._svc.sync(self._symbol, tf)
                if n:
                    self._log.info("candles_synced", tf=tf, new=n)
            except Exception as exc:
                self._log.warning("sync_failed", tf=tf, error=str(exc))

        # ── current price ──────────────────────────────────────────────────
        try:
            ticker = await self._client.fetch_ticker(self._symbol)
            price  = float(ticker["last"])
        except Exception as exc:
            self._log.error("ticker_failed", error=str(exc))
            return

        # ── load candles ───────────────────────────────────────────────────
        s        = get_settings()
        tact_df  = await self._load_df(self._signal_tf, limit=300)
        core_df  = await self._load_df(s.hybrid_long_signal_tf, limit=300)
        trend_df = await self._load_df(TREND_TF, limit=300)
        mid_df   = await self._load_df(s.hybrid_short_mid_tf, limit=300)

        # ── regime notification ────────────────────────────────────────────
        self._check_regime_notify(trend_df)

        # ── exits ──────────────────────────────────────────────────────────
        await self._manager.tick_exits(price)

        # ── tactical sleeve (every tick) ──────────────────────────────────
        await self._manager.tick_tactical(price, tact_df, trend_df, mid_df)

        # ── core sleeve (every 4h boundary) ───────────────────────────────
        if run_core:
            await self._manager.tick_core(price, core_df, trend_df)

        await self._log_portfolio(price)

    async def _load_df(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        now  = datetime.now(timezone.utc)
        days = {"15m": 10, "1h": 20, "4h": 60, "1d": 400}.get(timeframe, 30)
        since = now - timedelta(days=days)
        async with self._factory() as session:
            repo    = CandleRepository(session)
            candles = await repo.get_range(self._symbol, "gateio", timeframe, since, now)
        if not candles:
            return pd.DataFrame()
        rows = [
            {"time": c.time, "open": float(c.open), "high": float(c.high),
             "low": float(c.low), "close": float(c.close), "volume": float(c.volume)}
            for c in candles[-limit:]
        ]
        df = pd.DataFrame(rows).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _check_regime_notify(self, trend_df: pd.DataFrame) -> None:
        if trend_df.empty:
            return
        try:
            state = self._trend_eng.analyse(trend_df, self._symbol, TREND_TF)
        except Exception:
            return
        regime = state.regime
        self._log.info("regime_check", regime=str(regime))
        if regime in _NOTIFY_REGIMES and regime != self._last_regime:
            msg = f"smart-trader: {self._symbol} regime → {regime}"
            self._log.info("regime_changed_notify", regime=str(regime))
            try:
                subprocess.Popen([
                    "osascript", "-e",
                    f'display notification "{msg}" with title "smart-trader" sound name "Ping"',
                ])
            except Exception as exc:
                self._log.warning("notify_failed", error=str(exc))
        self._last_regime = regime

    async def _log_portfolio(self, price: float) -> None:
        p = await self._engine.get_portfolio(
            initial_cash=self._initial_cash,
            current_prices={self._symbol: price},
            peak_key=self._symbol,
        )
        self._log.info(
            "portfolio_snapshot",
            total=round(p.total_value, 2),
            cash=round(p.cash, 2),
            positions=len(p.open_positions),
            daily_pnl=round(p.daily_pnl, 4),
            drawdown=f"{p.drawdown_pct:.2%}",
        )

        # ── daily benchmark snapshot (once per calendar day) ──────────────
        today = datetime.now(timezone.utc).date()
        if self._benchmark_day != today:
            regime_str = str(self._last_regime) if self._last_regime else None
            try:
                now = datetime.now(timezone.utc)
                async with self._factory() as session:
                    repo = BenchmarkRepository(session)
                    await repo.ensure_baseline(
                        symbol=self._symbol,
                        start_price=price,
                        start_capital=self._initial_cash,
                        start_ts=now,
                    )
                    await repo.upsert_snapshot(
                        ts=now,
                        symbol=self._symbol,
                        portfolio_total=p.total_value,
                        cash=p.cash,
                        bh_price=price,
                        regime=regime_str,
                        positions=len(p.open_positions),
                    )
                self._benchmark_day = today
                self._log.info("benchmark_snapshot_saved", date=str(today),
                               total=round(p.total_value, 2), bh_price=price)
            except Exception as exc:
                self._log.warning("benchmark_snapshot_failed", error=str(exc))

    async def _sleep_until_next_candle(self, tick_start: datetime) -> None:
        tf_seconds = {
            "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
        }.get(self._signal_tf, 3600)
        now     = datetime.now(timezone.utc).timestamp()
        next_ts = (int(now / tf_seconds) + 1) * tf_seconds
        wake_ts = next_ts + 2
        self._log.info("sleeping", seconds=int(wake_ts - now))
        while True:
            remaining = wake_ts - datetime.now(timezone.utc).timestamp()
            if remaining <= 0:
                break
            await asyncio.sleep(min(EXIT_CHECK_INTERVAL, remaining))
            if datetime.now(timezone.utc).timestamp() >= wake_ts:
                break
            # mid-cycle: check SL/TP and time-stops between candle closes
            try:
                ticker = await self._client.fetch_ticker(self._symbol)
                await self._manager.tick_exits(float(ticker["last"]))
            except Exception as exc:
                self._log.warning("mid_cycle_check_failed", error=str(exc))
