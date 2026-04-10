"""WalkForwardOptimizer — Bayesian parameter search via Optuna.

Walk-forward scheme
────────────────────
  Total data: N days
  Train window: train_days  (e.g. 120)
  Test  window: test_days   (e.g. 60)
  Step : step_days          (e.g. 30)

  Window 1: train [0, 120),  test [120, 180)
  Window 2: train [30, 150), test [150, 210)   ← requires more data
  …

With only 180 days: 1 window.
With 1 year       : ~6 windows.

Objective
─────────
  score = sharpe × (1 − max_drawdown_pct / 100)

  Maximised by Optuna TPE sampler.  Minimum 5 trades required;
  otherwise the score is 0.0 (avoids degenerate "never trade" solutions).

Search space (10 params)
────────────────────────
  min_confidence        [0.55, 0.80]
  atr_mult              [1.2,  3.0 ]
  rr_ratio              [2.0,  5.0 ]
  kelly_frac            [0.15, 0.50]
  trailing_atr_mult     [1.0,  2.5 ]
  trailing_trigger_pct  [0.55, 0.90]
  breakeven_trigger_pct [0.30, 0.65]
  pullback_atr_frac     [0.20, 0.50]
  sl_cooldown_bars      {1 … 6}      (int)
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

# silence optuna's per-trial output — we handle logging ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)


@dataclass
class OptimizationWindow:
    train_start: datetime
    train_end:   datetime
    test_start:  datetime
    test_end:    datetime


@dataclass
class WindowResult:
    window:       OptimizationWindow
    best_params:  dict[str, Any]
    train_score:  float
    test_score:   float
    train_metrics: PerformanceMetrics
    test_metrics:  PerformanceMetrics


@dataclass
class OptimizationResult:
    symbol:       str
    timeframe:    str
    study_name:   str
    windows:      list[WindowResult] = field(default_factory=list)

    @property
    def best_params(self) -> dict[str, Any]:
        """Parameters from the window with the highest test score."""
        if not self.windows:
            return {}
        return max(self.windows, key=lambda w: w.test_score).best_params

    @property
    def avg_test_score(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.test_score for w in self.windows) / len(self.windows)


class WalkForwardOptimizer:
    """Bayesian walk-forward parameter optimiser.

    Usage::
        opt = WalkForwardOptimizer(symbol="ETH/USDT", timeframe="1h")
        result = opt.run(sig_df, trend_df, mid_df, n_trials=100)
    """

    def __init__(
        self,
        symbol:      str   = "BTC/USDT",
        timeframe:   str   = "1h",
        train_days:  int   = 120,
        test_days:   int   = 60,
        step_days:   int   = 30,
        min_trades:  int   = 5,
        strategy_version: str = "v2",
    ) -> None:
        self.symbol           = symbol
        self.timeframe        = timeframe
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
        mid_df:    pd.DataFrame | None = None,
        n_trials:  int = 100,
        study_name: str | None = None,
    ) -> OptimizationResult:
        """Run walk-forward optimisation, return results for all windows."""
        name = study_name or f"{self.symbol.replace('/', '_')}_{self.timeframe}"
        result = OptimizationResult(symbol=self.symbol, timeframe=self.timeframe, study_name=name)

        windows = self._make_windows(sig_df)
        if not windows:
            log.warning("Not enough data for any walk-forward window.")
            return result

        log.info(f"[{name}] {len(windows)} walk-forward window(s), {n_trials} trials each")

        for i, window in enumerate(windows):
            log.info(
                f"  Window {i+1}/{len(windows)}: "
                f"train {window.train_start.date()}→{window.train_end.date()}  "
                f"test  {window.test_start.date()}→{window.test_end.date()}"
            )
            wr = self._run_window(window, sig_df, trend_df, mid_df, n_trials, f"{name}_w{i+1}")
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

        # If we don't have enough for the requested split, scale down proportionally
        needed = self.train_days + self.test_days
        if total_days < needed:
            scale       = total_days / needed
            train_days  = max(30, int(self.train_days * scale))
            test_days   = max(10, total_days - train_days)
        else:
            train_days  = self.train_days
            test_days   = self.test_days

        windows = []
        offset  = pd.Timedelta(0)

        while True:
            train_start = start + offset
            train_end   = train_start + pd.Timedelta(days=train_days)
            test_start  = train_end
            test_end    = test_start + pd.Timedelta(days=test_days)

            if test_end > end + pd.Timedelta(days=1):
                break

            # clamp test_end to available data
            test_end = min(test_end, end + pd.Timedelta(hours=1))

            windows.append(OptimizationWindow(
                train_start = train_start.to_pydatetime().replace(tzinfo=timezone.utc),
                train_end   = train_end.to_pydatetime().replace(tzinfo=timezone.utc),
                test_start  = test_start.to_pydatetime().replace(tzinfo=timezone.utc),
                test_end    = test_end.to_pydatetime().replace(tzinfo=timezone.utc),
            ))
            offset += pd.Timedelta(days=self.step_days)

        return windows

    def _slice(self, df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        idx = pd.to_datetime(df.index, utc=True)
        mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
        return df.iloc[mask]

    def _run_window(
        self,
        window:    OptimizationWindow,
        sig_df:    pd.DataFrame,
        trend_df:  pd.DataFrame,
        mid_df:    pd.DataFrame | None,
        n_trials:  int,
        study_name: str,
    ) -> WindowResult:
        train_sig   = self._slice(sig_df,   window.train_start, window.train_end)
        train_trend = self._slice(trend_df, window.train_start, window.train_end)
        train_mid   = self._slice(mid_df,   window.train_start, window.train_end) if mid_df is not None else None

        test_sig    = self._slice(sig_df,   window.test_start, window.test_end)
        test_trend  = self._slice(trend_df, window.test_start, window.test_end)
        test_mid    = self._slice(mid_df,   window.test_start, window.test_end) if mid_df is not None else None

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest(trial)
            cfg    = self._make_config(params)
            m      = BacktestEngine(cfg).run(train_sig, train_trend, train_mid).metrics
            return self._score(m)

        def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            n     = len(study.trials)
            best  = study.best_value
            cur   = trial.value if trial.value is not None else 0.0
            print(
                f"  [{self.symbol}] trial {n:>3}/{n_trials}  "
                f"score={cur:.4f}  best={best:.4f}",
                flush=True,
            )

        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[_progress_callback])

        best_params    = study.best_params
        best_cfg       = self._make_config(best_params)
        train_metrics  = BacktestEngine(best_cfg).run(train_sig, train_trend, train_mid).metrics
        test_metrics   = BacktestEngine(best_cfg).run(test_sig,  test_trend,  test_mid).metrics

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
            "min_confidence":        trial.suggest_float("min_confidence",        0.55, 0.80),
            "atr_mult":              trial.suggest_float("atr_mult",              1.2,  3.0),
            "rr_ratio":              trial.suggest_float("rr_ratio",              2.0,  5.0),
            "kelly_frac":            trial.suggest_float("kelly_frac",            0.20, 0.70),
            "max_position_pct":      trial.suggest_float("max_position_pct",      0.04, 0.15),
            "trailing_atr_mult":     trial.suggest_float("trailing_atr_mult",     1.0,  2.5),
            "trailing_trigger_pct":  trial.suggest_float("trailing_trigger_pct",  0.55, 0.90),
            "breakeven_trigger_pct": trial.suggest_float("breakeven_trigger_pct", 0.30, 0.65),
            "pullback_atr_frac":     trial.suggest_float("pullback_atr_frac",     0.20, 0.50),
            "sl_cooldown_bars":      trial.suggest_int(  "sl_cooldown_bars",      1,    6),
            "min_conf_high_vol":     trial.suggest_float("min_conf_high_vol",     0.65, 0.85),
        }

    def _make_config(self, params: dict[str, Any]) -> BacktestConfig:
        return BacktestConfig(
            symbol           = self.symbol,
            timeframe        = self.timeframe,
            strategy_version = self.strategy_version,
            min_confidence         = params["min_confidence"],
            atr_mult               = params["atr_mult"],
            rr_ratio               = params["rr_ratio"],
            kelly_frac             = params["kelly_frac"],
            max_position_pct       = params["max_position_pct"],
            trailing_atr_mult      = params["trailing_atr_mult"],
            trailing_trigger_pct   = params["trailing_trigger_pct"],
            breakeven_trigger_pct  = params["breakeven_trigger_pct"],
            pullback_atr_frac      = params["pullback_atr_frac"],
            sl_cooldown_bars       = params["sl_cooldown_bars"],
            min_confidence_high_vol= params["min_conf_high_vol"],
        )

    def _score(self, m: PerformanceMetrics) -> float:
        """Objective: blend of Sharpe, annualised return, and drawdown penalty."""
        if m.total_trades < self.min_trades:
            return 0.0
        sharpe    = max(0.0, m.sharpe_ratio)
        ann_ret   = max(0.0, m.annualized_return)   # CAGR, [0, ∞)
        dd        = min(1.0, m.max_drawdown_pct / 100.0)
        # 60% Sharpe quality + 40% annualised return, both penalised by drawdown
        return (0.6 * sharpe + 0.4 * ann_ret * 10) * (1.0 - dd)
