"""TradingLoop — main async event loop.

Lifecycle per tick (every candle close):
  1.  sync_candles     — fetch new bars from exchange → DB
  2.  check_exits      — evaluate SL/TP on all open trades
  3.  analyse_trend    — TrendEngine on daily candles
  4.  mid_tf_direction — optional 4h trend alignment check
  5.  load_candles     — load signal/trend/mid DataFrames from DB
  6.  vol_regime       — VolatilityAnalyzer skip gate + source filter
  7.  detect_signals   — SignalEngine on hourly candles
  8.  cooldown_filter  — skip N bars after stop-loss
  9.  mtf_filter       — mid-tf direction alignment
  10. evaluate_risk    — RiskManager gate (vol-aware sizing)
  11. execute_entry    — place paper/live order, record trade
  12. log_state        — portfolio snapshot to log

Daily maintenance (first tick of the day):
  • reset circuit breaker
  • log daily P&L summary

Signal-quality guards (mirrors the improved backtest):
  • SL cooldown: after stop-loss, pause N bars before same-direction re-entry
  • Only one open position per symbol at a time
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import structlog

from smart_trader.core.settings import get_settings
from smart_trader.data.ingestion.candle_service import CandleIngestionService
from smart_trader.data.ingestion.gateio_client import GateIOClient
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.execution.engine import ExecutionEngine
from smart_trader.risk.manager import RiskManager
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import MarketRegime
from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer

log = structlog.get_logger(__name__)

# ── default configuration ─────────────────────────────────────────────────────
SYMBOL         = "BTC/USDT"
SIGNAL_TF      = "1h"
TREND_TF       = "1d"
MID_TF         = "4h"
INITIAL_CASH   = 10_000.0
MIN_CANDLES          = 60
SL_COOLDOWN          = 3     # bars to skip after stop-loss (normal vol) — overridden by OPT_SL_COOLDOWN_BARS
SL_COOLDOWN_HIGH_VOL = 8     # bars to skip after stop-loss in HIGH/CRISIS vol regime
MTF_THRESHOLD        = 0.10  # mid-tf direction threshold for MTF filter
MIN_CONF_HIGH_VOL    = 0.75  # raised confidence floor in HIGH/CRISIS vol regime
EXIT_CHECK_INTERVAL  = 300   # seconds — monitor SL/TP between candles (every 5 min)


class TradingLoop:
    """Paper-trading main loop.

    Usage::
        loop = TradingLoop(paper=True)
        await loop.run()          # blocks until KeyboardInterrupt
    """

    def __init__(
        self,
        symbol:          str                                        = SYMBOL,
        signal_tf:       str                                        = SIGNAL_TF,
        trend_tf:        str                                        = TREND_TF,
        mid_tf:          str                                        = MID_TF,
        initial_cash:    float                                      = INITIAL_CASH,
        paper:           bool                                       = True,
        signal_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.symbol          = symbol
        self.signal_tf       = signal_tf
        self.trend_tf        = trend_tf
        self.mid_tf          = mid_tf
        self.initial_cash    = initial_cash
        self.paper           = paper
        self._signal_callback = signal_callback

        # bind logger first so it's available immediately
        self._log = log.bind(symbol=symbol, paper=paper)

        s = get_settings()
        api_key    = s.gateio_api_key.get_secret_value()
        api_secret = s.gateio_api_secret.get_secret_value()

        self._client   = GateIOClient(api_key=api_key, api_secret=api_secret, paper=paper)
        self._svc      = CandleIngestionService(self._client)
        self._factory  = get_session_factory()
        self._engine   = ExecutionEngine(client=self._client, paper=paper)
        self._risk     = RiskManager.from_settings()
        self._trend    = TrendEngine()
        self._signal   = SignalEngine(version=s.strategy_version)
        self._vol      = VolatilityAnalyzer()
        # use optimised cooldown if set in .env
        self._sl_cooldown_bars = s.opt_sl_cooldown_bars if s.opt_sl_cooldown_bars is not None else SL_COOLDOWN
        self._log.info("strategy_version", version=s.strategy_version,
                       sl_cooldown=self._sl_cooldown_bars,
                       atr_mult=s.opt_atr_mult, rr_ratio=s.opt_rr_ratio)

        # state
        self._last_day:         Optional[date] = None
        self._sl_cooldown:      dict[str, int] = {"buy": 0, "sell": 0}
        self._sl_vol_regime:    dict[str, str] = {"buy": "", "sell": ""}  # vol regime at entry
        self._cur_vol_regime:   str = ""  # current vol regime (updated each tick)
        self._consecutive_bear: int = 0          # consecutive bearish days (direction-based)
        self._last_bear_date:  Optional[date] = None  # dedup by calendar day

    # ── public ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Block forever, running a full tick at each candle close.

        Between candle closes, wakes every EXIT_CHECK_INTERVAL seconds to
        evaluate SL/TP on open positions so stops trigger promptly.
        The first tick runs immediately on startup without waiting.
        """
        self._log.info("loop_started", signal_tf=self.signal_tf, trend_tf=self.trend_tf)
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
            self._log.info("loop_stopped")

    async def run_once(self) -> None:
        """Run a single tick (useful for testing / manual trigger)."""
        try:
            await self._tick()
        finally:
            await self._client.close()

    # ── private: main tick ────────────────────────────────────────────────────

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        self._log.info("tick_begin", time=now.isoformat())

        # per-tick record — emitted to signal_callback at every exit path
        tick: dict[str, Any] = {
            "time":              now.isoformat(),
            "symbol":            self.symbol,
            "price":             0.0,
            "regime":            None,
            "mid_direction":     None,
            "vol_regime":        None,
            "vol_state":         None,
            "vol_skip":          False,
            "signals":           [],
            "decision":          "hold",
            "decision_reason":   None,
            "entry_side":        None,
            "entry_confidence":  None,
        }

        def _emit() -> None:
            if self._signal_callback:
                self._signal_callback(dict(tick))

        # ── daily maintenance ─────────────────────────────────────────────
        today = now.date()
        if self._last_day != today:
            self._risk.reset_breaker()
            self._log.info("circuit_breaker_reset", date=str(today))
            self._last_day = today

        # ── 1. sync candles ───────────────────────────────────────────────
        for tf in (self.signal_tf, self.trend_tf, self.mid_tf):
            try:
                n = await self._svc.sync(self.symbol, tf)
                if n:
                    self._log.info("candles_synced", tf=tf, new=n)
            except Exception as exc:
                self._log.warning("sync_failed", tf=tf, error=str(exc))

        # ── 2. check open trade exits (SL / TP) ───────────────────────────
        try:
            ticker = await self._client.fetch_ticker(self.symbol)
            price  = float(ticker["last"])
            tick["price"] = price
        except Exception as exc:
            self._log.error("ticker_failed", error=str(exc))
            return

        closed = await self._engine.check_exits({self.symbol: price})
        for ct in closed:
            self._log.info(
                "trade_closed",
                trade_id=str(ct.trade_id)[:8],
                reason=ct.exit_reason,
                pnl=ct.pnl,
                pnl_pct=f"{ct.pnl_pct:.2%}",
            )
            side = "buy" if ct.side == "buy" else "sell"
            if ct.exit_reason == "stop_loss":
                # adaptive cooldown: longer pause after stop in HIGH/CRISIS vol
                _vol_at_entry = self._sl_vol_regime.get(side, "")
                bars = SL_COOLDOWN_HIGH_VOL if _vol_at_entry in ("high", "crisis") else self._sl_cooldown_bars
                self._sl_cooldown[side] = bars
                self._log.info("sl_cooldown_set", side=side, bars=bars,
                               vol_regime=_vol_at_entry)

        # tick down cooldowns
        for s in self._sl_cooldown:
            if self._sl_cooldown[s] > 0:
                self._sl_cooldown[s] -= 1

        # ── 3. load candles for analysis ──────────────────────────────────
        sig_df   = await self._load_df(self.signal_tf, limit=300)
        trend_df = await self._load_df(self.trend_tf,  limit=300)
        mid_df   = await self._load_df(self.mid_tf,    limit=300)

        if len(sig_df) < MIN_CANDLES or len(trend_df) < MIN_CANDLES:
            self._log.warning("insufficient_candles",
                              sig=len(sig_df), trend=len(trend_df))
            tick["decision"] = "insufficient_data"
            _emit()
            return

        # ── 4. trend analysis ─────────────────────────────────────────────
        try:
            trend_state = self._trend.analyse(trend_df, self.symbol, self.trend_tf)
            tick["regime"] = str(trend_state.regime)
            # bear market filter counter (count calendar days, not ticks)
            # Uses direction threshold so BEAR_RANGING / DISTRIBUTION also count.
            _bear_threshold = getattr(get_settings(), "bear_filter_threshold", -0.15)
            _today = now.date()
            if trend_state.direction < _bear_threshold:
                if self._last_bear_date is None or _today != self._last_bear_date:
                    self._consecutive_bear += 1
                    self._last_bear_date = _today
            else:
                self._consecutive_bear = 0
                self._last_bear_date   = None
            self._log.info(
                "trend_analysed",
                regime=str(trend_state.regime),
                direction=trend_state.direction,
                strength=trend_state.strength,
                consecutive_bear=self._consecutive_bear,
            )
        except Exception as exc:
            self._log.error("trend_failed", error=str(exc))
            tick["decision"] = "trend_error"
            _emit()
            return

        # ── 5. mid-tf direction ───────────────────────────────────────────
        mid_direction: Optional[float] = None
        if len(mid_df) >= MIN_CANDLES:
            try:
                mid_state = self._trend.analyse(mid_df, self.symbol, self.mid_tf)
                mid_direction = mid_state.direction
                tick["mid_direction"] = round(mid_direction, 4)
            except Exception:
                pass

        # ── 6. volatility regime ──────────────────────────────────────────
        vol_state = None
        try:
            vol_state = self._vol.analyse(trend_df, sig_df)
            tick["vol_regime"] = str(vol_state.regime)
            tick["vol_state"]  = str(vol_state.state)
            self._log.info(
                "vol_regime",
                regime=str(vol_state.regime),
                state=str(vol_state.state),
                hist_vol=f"{vol_state.hist_vol_annual:.0%}",
                atr_rank=f"{vol_state.atr_rank:.0%}",
                vol_ratio=round(vol_state.vol_ratio, 2),
                atr_mult=vol_state.atr_mult,
                kelly_scale=vol_state.kelly_scale,
                skip=vol_state.skip,
            )
        except Exception as exc:
            self._log.warning("vol_analysis_failed", error=str(exc))

        if vol_state is not None and vol_state.skip:
            self._log.info(
                "vol_skip",
                regime=str(vol_state.regime),
                state=str(vol_state.state),
            )
            tick["decision"] = "vol_skip"
            tick["vol_skip"] = True
            _emit()
            await self._log_portfolio(price)
            return

        # persist current vol regime for adaptive cooldown on future stops
        self._cur_vol_regime = str(vol_state.regime) if vol_state is not None else ""

        # ── 7. signal detection ───────────────────────────────────────────
        s = get_settings()
        _is_high_vol = self._cur_vol_regime in ("high", "crisis")
        _min_conf    = MIN_CONF_HIGH_VOL if _is_high_vol else s.confidence_threshold
        signals = self._signal.analyse(
            sig_df,
            symbol=self.symbol,
            timeframe=self.signal_tf,
            trend_state=trend_state,
            vol_state=vol_state,
            min_confidence=_min_conf,
        )

        # capture all signals (before vol-source filter) in tick record
        tick["signals"] = [
            {
                "source":      sg.source,
                "signal_type": sg.signal_type,
                "raw_score":   round(sg.raw_score, 4),
                "confidence":  round(sg.confidence, 4),
                "features":    {k: round(v, 4) for k, v in sg.features.items()},
            }
            for sg in signals
        ]

        if not signals:
            self._log.debug("no_signals")
            tick["decision"] = "no_signals"
            _emit()
            await self._log_portfolio(price)
            return

        # filter by preferred signal sources when vol regime is known
        if vol_state is not None and vol_state.preferred_sources:
            signals = [
                sg for sg in signals
                if any(src in sg.source for src in vol_state.preferred_sources)
            ]
            if not signals:
                self._log.debug("no_signals_after_vol_filter")
                tick["decision"] = "vol_source_filtered"
                _emit()
                await self._log_portfolio(price)
                return

        best = signals[0]
        entry_side = "buy" if best.signal_type == "long" else "sell"
        self._log.info(
            "signal_detected",
            type=best.signal_type,
            source=best.source,
            confidence=best.confidence,
            regime=str(best.regime),
        )

        # ── 8a. bear market filter ───────────────────────────────────────
        _bear_bars = getattr(get_settings(), "bear_filter_bars", 3)
        if self._consecutive_bear >= _bear_bars and entry_side == "buy":
            self._log.info("bear_filter_skip",
                           consecutive_bear=self._consecutive_bear,
                           threshold=_bear_bars)
            tick["decision"]        = "mtf_blocked"
            tick["decision_reason"] = f"bear_trending {self._consecutive_bear} bars"
            tick["entry_side"]      = entry_side
            _emit()
            await self._log_portfolio(price)
            return

        # ── 8b. cooldown filter ───────────────────────────────────────────
        if self._sl_cooldown[entry_side] > 0:
            self._log.info("cooldown_skip",
                           side=entry_side, remaining=self._sl_cooldown[entry_side])
            tick["decision"]        = "cooldown"
            tick["decision_reason"] = f"{self._sl_cooldown[entry_side]} bars remaining"
            tick["entry_side"]      = entry_side
            _emit()
            await self._log_portfolio(price)
            return

        # ── 9. MTF confirmation ───────────────────────────────────────────
        if mid_direction is not None:
            if best.signal_type == "long"  and mid_direction < -MTF_THRESHOLD:
                self._log.info("mtf_blocked", signal="long",  mid_dir=mid_direction)
                tick["decision"]        = "mtf_blocked"
                tick["decision_reason"] = f"4h direction={mid_direction:.3f}"
                tick["entry_side"]      = entry_side
                _emit()
                await self._log_portfolio(price)
                return
            if best.signal_type == "short" and mid_direction >  MTF_THRESHOLD:
                self._log.info("mtf_blocked", signal="short", mid_dir=mid_direction)
                tick["decision"]        = "mtf_blocked"
                tick["decision_reason"] = f"4h direction={mid_direction:.3f}"
                tick["entry_side"]      = entry_side
                _emit()
                await self._log_portfolio(price)
                return

        # ── 10. risk evaluation ───────────────────────────────────────────
        portfolio = await self._engine.get_portfolio(
            initial_cash=self.initial_cash,
            current_prices={self.symbol: price},
        )
        decision = self._risk.evaluate(best, portfolio, price, sig_df, trend_df)

        if not decision.approved:
            self._log.info("risk_rejected", reasons=decision.reasons)
            tick["decision"]        = "risk_rejected"
            tick["decision_reason"] = "; ".join(decision.reasons)
            tick["entry_side"]      = entry_side
            _emit()
            await self._log_portfolio(price)
            return

        # ── 11. execute entry ─────────────────────────────────────────────
        # record vol regime so adaptive cooldown uses correct length if stopped
        self._sl_vol_regime[entry_side] = self._cur_vol_regime

        trade_id = await self._engine.process_signal(best, decision, price)
        if trade_id:
            pos = decision.position_size
            self._log.info(
                "trade_opened",
                trade_id=str(trade_id)[:8],
                side=entry_side,
                price=price,
                qty=pos.quantity,
                sl=pos.stop_price,
                tp=pos.take_profit_price,
                confidence=best.confidence,
                vol_regime=self._cur_vol_regime,
            )

        tick["decision"]          = "enter"
        tick["entry_side"]        = entry_side
        tick["entry_confidence"]  = round(best.confidence, 4)
        _emit()
        await self._log_portfolio(price)

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _load_df(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Load most-recent `limit` candles from DB into a DataFrame."""
        now = datetime.now(timezone.utc)
        # generous look-back window to cover `limit` bars for each timeframe
        days = {"1h": 20, "4h": 60, "1d": 400}.get(timeframe, 30)
        since = now - timedelta(days=days)

        async with self._factory() as session:
            repo    = CandleRepository(session)
            candles = await repo.get_range(
                self.symbol, "gateio", timeframe, since, now
            )

        if not candles:
            return pd.DataFrame()

        rows = [
            {
                "time":   c.time,
                "open":   float(c.open),
                "high":   float(c.high),
                "low":    float(c.low),
                "close":  float(c.close),
                "volume": float(c.volume),
            }
            for c in candles[-limit:]
        ]
        df = pd.DataFrame(rows).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def _log_portfolio(self, price: float) -> None:
        portfolio = await self._engine.get_portfolio(
            initial_cash=self.initial_cash,
            current_prices={self.symbol: price},
        )
        self._log.info(
            "portfolio_snapshot",
            total=round(portfolio.total_value, 2),
            cash=round(portfolio.cash, 2),
            open_positions=list(portfolio.open_positions.keys()),
            daily_pnl=round(portfolio.daily_pnl, 4),
            drawdown=f"{portfolio.drawdown_pct:.2%}",
        )

    async def _sleep_until_next_candle(self, tick_start: datetime) -> None:
        """Sleep until the next candle close, checking exits every EXIT_CHECK_INTERVAL seconds."""
        tf_seconds = {
            "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }.get(self.signal_tf, 3600)

        now     = datetime.now(timezone.utc).timestamp()
        next_ts = (int(now / tf_seconds) + 1) * tf_seconds
        wake_ts = next_ts + 2   # +2 s buffer for exchange lag

        self._log.info(
            "sleeping",
            seconds=int(wake_ts - now),
            next_candle=datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat(),
            exit_check_every=EXIT_CHECK_INTERVAL,
        )

        while True:
            remaining = wake_ts - datetime.now(timezone.utc).timestamp()
            if remaining <= 0:
                break
            await asyncio.sleep(min(EXIT_CHECK_INTERVAL, remaining))

            # re-check: did the sleep overshoot the target?
            if datetime.now(timezone.utc).timestamp() >= wake_ts:
                break

            # mid-cycle: only check SL/TP on open positions
            await self._mid_cycle_exit_check()

    async def _mid_cycle_exit_check(self) -> None:
        """Check open-position exits between candle closes (SL/TP only, no new entries)."""
        try:
            ticker = await self._client.fetch_ticker(self.symbol)
            price  = float(ticker["last"])
        except Exception as exc:
            self._log.warning("mid_cycle_ticker_failed", error=str(exc))
            return

        try:
            closed = await self._engine.check_exits({self.symbol: price})
        except Exception as exc:
            self._log.warning("mid_cycle_exits_failed", error=str(exc))
            return

        for ct in closed:
            self._log.info(
                "trade_closed",
                trade_id=str(ct.trade_id)[:8],
                reason=ct.exit_reason,
                pnl=ct.pnl,
                pnl_pct=f"{ct.pnl_pct:.2%}",
                mid_cycle=True,
            )
            side = "buy" if ct.side == "buy" else "sell"
            if ct.exit_reason == "stop_loss":
                _vol_at_entry = self._sl_vol_regime.get(side, "")
                bars = SL_COOLDOWN_HIGH_VOL if _vol_at_entry in ("high", "crisis") else self._sl_cooldown_bars
                self._sl_cooldown[side] = bars
                self._log.info("sl_cooldown_set", side=side, bars=bars,
                               vol_regime=_vol_at_entry)
