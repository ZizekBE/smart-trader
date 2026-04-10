"""CoreSleeveOptimizer — Bayesian parameter search for the core (4h) sleeve.

Walk-forward scheme
────────────────────
  Mirrors WalkForwardOptimizer but tuned for 4h signal frequency:
  • Signal TF  : 4h  (fewer bars per period — windows are smaller)
  • Trend TF   : 1d
  • No mid_df  : require_mtf=False (core sleeve doesn't use MTF filter)
  • Min trades : 3   (4h signals are rarer — don't over-penalise)

Objective
─────────
  score = sharpe × (1 − max_drawdown_pct / 100)
  Same as WalkForwardOptimizer — maximised by Optuna TPE.

Search space (adjusted for core sleeve)
────────────────────────────────────────
  min_confidence        [0.55, 0.75]   (tighter: core can be more selective)
  atr_mult              [1.5,  4.0 ]   (wider SL for swing positions)
  rr_ratio              [1.5,  4.0 ]   (allow lower R:R for more trades)
  kelly_frac            [0.15, 0.40]
  trailing_atr_mult     [1.0,  3.0 ]
  trailing_trigger_pct  [0.55, 0.90]
  breakeven_trigger_pct [0.30, 0.65]
  sl_cooldown_bars      {1 … 4}        (4h bars; shorter lookback than 1h)
  min_conf_high_vol     [0.65, 0.85]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import optuna
import pandas as pd

from smart_trader.analysis.backtest.engine import BacktestConfig, BacktestEngine
from smart_trader.analysis.metrics.calculator import PerformanceMetrics
from smart_trader.analysis.optimizer.study import OptimizationResult, OptimizationWindow, WindowResult

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)


class CoreSleeveOptimizer:
    """Bayesian walk-forward optimiser for the core (4h) sleeve.

    Usage::
        opt    = CoreSleeveOptimizer(symbol="BTC/USDT")
        result = opt.run(sig_4h_df, trend_1d_df, n_trials=100)
    """

    def __init__(
        self,
        symbol:      str  = "BTC/USDT",
        train_days:  int  = 120,
        test_days:   int  = 60,
        step_days:   int  = 30,
        min_trades:  int  = 3,
        strategy_version: str = "v2",
    ) -> None:
        self.symbol           = symbol
        self.timeframe        = "4h"
        self.train_days       = train_days
        self.test_days        = test_days
        self.step_days        = step_days
        self.min_trades       = min_trades
        self.strategy_version = strategy_version

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        sig_df:    pd.DataFrame,
        trend_df:  pd.DataFrame,
        n_trials:  int = 100,
        study_name: str | None = None,
    ) -> OptimizationResult:
        """Run walk-forward optimisation on 4h signal data."""
        name   = study_name or f"{self.symbol.replace('/', '_')}_4h_core"
        result = OptimizationResult(symbol=self.symbol, timeframe=self.timeframe, study_name=name)

        windows = self._make_windows(sig_df)
        if not windows:
            log.warning("Not enough 4h data for any walk-forward window.")
            return result

        log.info(f"[{name}] {len(windows)} walk-forward window(s), {n_trials} trials each")

        for i, window in enumerate(windows):
            log.info(
                f"  Window {i+1}/{len(windows)}: "
                f"train {window.train_start.date()}→{window.train_end.date()}  "
                f"test  {window.test_start.date()}→{window.test_end.date()}"
            )
            wr = self._run_window(window, sig_df, trend_df, n_trials, f"{name}_w{i+1}")
            result.windows.append(wr)
            log.info(
                f"    best params: {wr.best_params}  "
                f"train={wr.train_score:.3f}  test={wr.test_score:.3f}"
            )

        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _make_windows(self, sig_df: pd.DataFrame) -> list[OptimizationWindow]:
        if sig_df.empty:
            return []

        idx   = sig_df.index
        start = pd.Timestamp(idx.min())
        end   = pd.Timestamp(idx.max())
        total_days = (end - start).days

        needed = self.train_days + self.test_days
        if total_days < needed:
            scale      = total_days / needed
            train_days = max(30, int(self.train_days * scale))
            test_days  = max(10, total_days - train_days)
        else:
            train_days = self.train_days
            test_days  = self.test_days

        windows = []
        offset  = pd.Timedelta(0)

        while True:
            train_start = start + offset
            train_end   = train_start + pd.Timedelta(days=train_days)
            test_start  = train_end
            test_end    = test_start + pd.Timedelta(days=test_days)

            if test_end > end + pd.Timedelta(days=1):
                break

            test_end = min(test_end, end + pd.Timedelta(hours=4))

            windows.append(OptimizationWindow(
                train_start = train_start.to_pydatetime().replace(tzinfo=timezone.utc),
                train_end   = train_end.to_pydatetime().replace(tzinfo=timezone.utc),
                test_start  = test_start.to_pydatetime().replace(tzinfo=timezone.utc),
                test_end    = test_end.to_pydatetime().replace(tzinfo=timezone.utc),
            ))
            offset += pd.Timedelta(days=self.step_days)

        return windows

    def _slice(self, df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        idx  = pd.to_datetime(df.index, utc=True)
        mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
        return df.iloc[mask]

    def _run_window(
        self,
        window:    OptimizationWindow,
        sig_df:    pd.DataFrame,
        trend_df:  pd.DataFrame,
        n_trials:  int,
        study_name: str,
    ) -> WindowResult:
        train_sig   = self._slice(sig_df,   window.train_start, window.train_end)
        train_trend = self._slice(trend_df, window.train_start, window.train_end)
        test_sig    = self._slice(sig_df,   window.test_start,  window.test_end)
        test_trend  = self._slice(trend_df, window.test_start,  window.test_end)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest(trial)
            cfg    = self._make_config(params)
            m      = BacktestEngine(cfg).run(train_sig, train_trend, None).metrics
            return self._score(m)

        def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            n    = len(study.trials)
            best = study.best_value
            cur  = trial.value if trial.value is not None else 0.0
            print(
                f"  [{self.symbol}] trial {n:>3}/{n_trials}  "
                f"score={cur:.4f}  best={best:.4f}",
                flush=True,
            )

        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[_progress_callback])

        best_params   = study.best_params
        best_cfg      = self._make_config(best_params)
        train_metrics = BacktestEngine(best_cfg).run(train_sig, train_trend, None).metrics
        test_metrics  = BacktestEngine(best_cfg).run(test_sig,  test_trend,  None).metrics

        return WindowResult(
            window        = window,
            best_params   = best_params,
            train_score   = self._score(train_metrics),
            test_score    = self._score(test_metrics),
            train_metrics = train_metrics,
            test_metrics  = test_metrics,
        )

    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "min_confidence":        trial.suggest_float("min_confidence",        0.55, 0.75),
            "atr_mult":              trial.suggest_float("atr_mult",              1.5,  4.0),
            "rr_ratio":              trial.suggest_float("rr_ratio",              1.5,  4.0),
            "kelly_frac":            trial.suggest_float("kelly_frac",            0.20, 0.70),
            "max_position_pct":      trial.suggest_float("max_position_pct",      0.04, 0.15),
            "trailing_atr_mult":     trial.suggest_float("trailing_atr_mult",     1.0,  3.0),
            "trailing_trigger_pct":  trial.suggest_float("trailing_trigger_pct",  0.55, 0.90),
            "breakeven_trigger_pct": trial.suggest_float("breakeven_trigger_pct", 0.30, 0.65),
            "sl_cooldown_bars":      trial.suggest_int(  "sl_cooldown_bars",      1,    4),
            "min_conf_high_vol":     trial.suggest_float("min_conf_high_vol",     0.65, 0.85),
        }

    def _make_config(self, params: dict[str, Any]) -> BacktestConfig:
        return BacktestConfig(
            symbol           = self.symbol,
            timeframe        = "4h",
            trend_timeframe  = "1d",
            strategy_version = self.strategy_version,
            require_mtf      = False,
            min_confidence         = params["min_confidence"],
            atr_mult               = params["atr_mult"],
            rr_ratio               = params["rr_ratio"],
            kelly_frac             = params["kelly_frac"],
            max_position_pct       = params["max_position_pct"],
            trailing_atr_mult      = params["trailing_atr_mult"],
            trailing_trigger_pct   = params["trailing_trigger_pct"],
            breakeven_trigger_pct  = params["breakeven_trigger_pct"],
            sl_cooldown_bars       = params["sl_cooldown_bars"],
            min_confidence_high_vol= params["min_conf_high_vol"],
        )

    def _score(self, m: PerformanceMetrics) -> float:
        if m.total_trades < self.min_trades:
            return 0.0
        sharpe  = max(0.0, m.sharpe_ratio)
        ann_ret = max(0.0, m.annualized_return)
        dd      = min(1.0, m.max_drawdown_pct / 100.0)
        return (0.6 * sharpe + 0.4 * ann_ret * 10) * (1.0 - dd)
