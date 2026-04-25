"""HybridBacktestEngine — run core + tactical sleeves in parallel backtests.

Approach
────────
Run two independent BacktestEngine instances (one per sleeve), each with its
own signal timeframe, confidence threshold and ATR/RR parameters.  After both
finish, merge all trades, recompute combined metrics on the merged equity curve,
and return a HybridBacktestResult with per-sleeve and combined numbers.

Capital split (mirrors the live hybrid loop):
  • Core     sleeve — 60 % of initial capital, max 30 % per position
  • Tactical sleeve — 30 % of initial capital, max 10 % per position

Because each BacktestEngine sees its own slice of capital, position sizing is
automatically scaled to the right dollar amounts.  Combined P&L is the sum of
both sleeves' P&L, re-evaluated against the full initial capital.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from smart_trader.analysis.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
)
from smart_trader.analysis.metrics.calculator import MetricsCalculator, PerformanceMetrics, TradeResult
from smart_trader.core.settings import get_settings


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SleeveResult:
    """Per-sleeve backtest result with sleeve label on each trade."""
    sleeve:  str              # 'core' | 'tactical'
    trades:  list[BacktestTrade]
    metrics: PerformanceMetrics
    config:  BacktestConfig


@dataclass
class HybridBacktestResult:
    core:     SleeveResult
    tactical: SleeveResult
    combined: PerformanceMetrics          # all trades, full initial capital
    all_trades: list[BacktestTrade]       # merged, sorted by entry time
    initial_capital: float
    start_at: Optional[datetime] = None
    end_at:   Optional[datetime] = None

    @property
    def long_trades(self) -> list[BacktestTrade]:
        return [t for t in self.all_trades if t.side == "buy"]

    @property
    def short_trades(self) -> list[BacktestTrade]:
        return [t for t in self.all_trades if t.side == "sell"]


# ── engine ────────────────────────────────────────────────────────────────────

class HybridBacktestEngine:
    """Run a dual-sleeve hybrid backtest from pre-loaded DataFrames."""

    def __init__(
        self,
        symbol:          str   = "BTC/USDT",
        initial_capital: float = 10_000.0,
        strategy_version: str  = "v2",
        regime_routing:   bool = False,
        long_only:        bool = False,
        # Per-symbol optimised core params (from optimization_runs table).
        # Takes priority over global settings and individual overrides.
        core_params:      Optional[dict[str, Any]] = None,
        # Core sleeve individual overrides (None = use settings)
        core_atr_mult:    Optional[float] = None,
        core_rr_ratio:    Optional[float] = None,
        core_kelly_frac:  Optional[float] = None,
        core_min_conf:    Optional[float] = None,
        # Tactical sleeve overrides (None = use settings / opt_* values)
        tact_atr_mult:    Optional[float] = None,
        tact_rr_ratio:    Optional[float] = None,
        tact_kelly_frac:  Optional[float] = None,
        tact_min_conf:    Optional[float] = None,
    ) -> None:
        s = get_settings()

        self._symbol          = symbol
        self._initial_capital = initial_capital
        self._regime_routing  = regime_routing
        self._long_only       = long_only

        long_budget  = initial_capital * s.hybrid_long_budget_pct
        short_budget = initial_capital * s.hybrid_short_budget_pct

        # ── resolve core params (per-symbol > individual override > settings) ─
        _cp = core_params or {}
        def _c(key: str, override: Optional[float], default: float) -> float:
            return float(_cp[key]) if key in _cp else (override if override is not None else default)

        # ── core sleeve config ────────────────────────────────────────────
        self._core_cfg = BacktestConfig(
            symbol=symbol,
            timeframe=s.hybrid_long_signal_tf,    # 4h
            trend_timeframe=s.hybrid_long_trend_tf,  # 1d
            initial_capital=long_budget,
            min_confidence=_c("min_confidence", core_min_conf, s.hybrid_long_min_conf),
            max_position_pct=s.hybrid_long_max_pos_pct,
            regime_routing=regime_routing,
            strategy_version=strategy_version,
            require_mtf=False,
            sl_cooldown_bars=int(_cp.get("sl_cooldown_bars", 4)),
            atr_mult=_c("atr_mult",   core_atr_mult,   s.hybrid_long_atr_mult),
            rr_ratio=_c("rr_ratio",   core_rr_ratio,   s.hybrid_long_rr_ratio),
            kelly_frac=_c("kelly_frac", core_kelly_frac, 0.25) if (core_kelly_frac or "kelly_frac" in _cp) else None,
            trailing_atr_mult=float(_cp["trailing_atr_mult"]) if "trailing_atr_mult" in _cp else 1.5,
            trailing_trigger_pct=float(_cp["trailing_trigger_pct"]) if "trailing_trigger_pct" in _cp else 0.75,
            breakeven_trigger_pct=float(_cp["breakeven_trigger_pct"]) if "breakeven_trigger_pct" in _cp else 0.50,
            min_confidence_high_vol=float(_cp["min_conf_high_vol"]) if "min_conf_high_vol" in _cp else s.hybrid_long_min_conf + 0.10,
            # features
            partial_tp=s.hybrid_partial_tp,
            partial_tp_r=s.hybrid_partial_tp_r,
            partial_tp_pct=s.hybrid_partial_tp_pct,
            long_only=long_only,
        )

        # ── tactical sleeve config ────────────────────────────────────────
        self._tact_cfg = BacktestConfig(
            symbol=symbol,
            timeframe=s.hybrid_short_signal_tf,    # 1h
            trend_timeframe=s.hybrid_short_trend_tf,  # 1d
            initial_capital=short_budget,
            min_confidence=tact_min_conf or s.hybrid_short_min_conf,
            max_position_pct=s.hybrid_short_max_pos_pct,  # 10 % of budget per trade
            strategy_version=strategy_version,
            regime_routing=regime_routing,
            require_mtf=True,        # tactical uses 4h MTF alignment
            sl_cooldown_bars=s.opt_sl_cooldown_bars or 3,
            atr_mult=tact_atr_mult  or s.opt_atr_mult,
            rr_ratio=tact_rr_ratio  or s.opt_rr_ratio,
            kelly_frac=tact_kelly_frac or s.opt_kelly_frac,
            # features
            session_filter=s.hybrid_session_filter,
            session_start_utc=s.hybrid_session_start_utc,
            session_end_utc=s.hybrid_session_end_utc,
            partial_tp=s.hybrid_partial_tp,
            partial_tp_r=s.hybrid_partial_tp_r,
            partial_tp_pct=s.hybrid_partial_tp_pct,
            long_only=long_only,
        )

        self._calc = MetricsCalculator()

    def run(
        self,
        h1_df:    pd.DataFrame,           # 1h candles  (tactical signal TF)
        h4_df:    pd.DataFrame,           # 4h candles  (core signal TF + tactical mid TF)
        d1_df:    pd.DataFrame,           # 1d candles  (shared trend TF)
    ) -> HybridBacktestResult:
        """Run both sleeves and return combined results.

        Args:
            h1_df:  1h OHLCV DataFrame (index = DatetimeIndex UTC)
            h4_df:  4h OHLCV DataFrame
            d1_df:  1d OHLCV DataFrame
        """
        # ── core sleeve: 4h signal, 1d trend, no MTF ─────────────────────
        core_engine = BacktestEngine(self._core_cfg)
        core_result = core_engine.run(
            signal_candles=h4_df,
            trend_candles=d1_df,
            mid_candles=None,     # core sleeve doesn't use MTF
        )

        # ── tactical sleeve: 1h signal, 1d trend, 4h MTF ─────────────────
        tact_engine = BacktestEngine(self._tact_cfg)
        tact_result = tact_engine.run(
            signal_candles=h1_df,
            trend_candles=d1_df,
            mid_candles=h4_df,
        )

        # ── merge & label ─────────────────────────────────────────────────
        core_trades = self._label(core_result.trades, "core")
        tact_trades = self._label(tact_result.trades, "tactical")

        all_trades = sorted(
            core_trades + tact_trades,
            key=lambda t: t.entry_time,
        )

        # ── combined metrics on full initial capital ───────────────────────
        combined_trade_results = [
            TradeResult(
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                opened_at=t.entry_time,
                closed_at=t.exit_time,
                fees=t.fees,
                slippage=t.slippage,
                side=t.side,
            )
            for t in all_trades
        ]
        combined_metrics = self._calc.calculate(combined_trade_results, self._initial_capital)

        # ── date range ────────────────────────────────────────────────────
        all_entries = [t.entry_time for t in all_trades]
        all_exits   = [t.exit_time  for t in all_trades]
        start_at = min(all_entries) if all_entries else None
        end_at   = max(all_exits)   if all_exits   else None

        return HybridBacktestResult(
            core=SleeveResult(
                sleeve="core",
                trades=core_trades,
                metrics=core_result.metrics,
                config=self._core_cfg,
            ),
            tactical=SleeveResult(
                sleeve="tactical",
                trades=tact_trades,
                metrics=tact_result.metrics,
                config=self._tact_cfg,
            ),
            combined=combined_metrics,
            all_trades=all_trades,
            initial_capital=self._initial_capital,
            start_at=start_at,
            end_at=end_at,
        )

    @staticmethod
    def _label(trades: list[BacktestTrade], sleeve: str) -> list[BacktestTrade]:
        """Stamp sleeve name into BacktestTrade.regime field for UI display."""
        out = []
        for t in trades:
            # Prepend sleeve tag to regime string so UI can distinguish
            import copy
            t2 = copy.copy(t)
            t2.regime = f"[{sleeve}] {t.regime}"
            out.append(t2)
        return out
