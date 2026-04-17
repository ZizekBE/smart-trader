#!/usr/bin/env python3
"""Phase 2.3 strategy comparison — walk-forward evaluation.

Compares four signal-selection strategies across 20 sequential 7-day folds:

  v2_all          — baseline: all 4 detectors (current v2)
  trend_follower  — macd + ema_bounce only
  mean_reversion  — rsi + bollinger only
  routed_v23      — Phase 2.3 routing: trending→trend_follower,
                    ranging→mean_reversion, other→all

The comparison uses the labeled parquet produced by build_signal_labels.py.
Source field in the parquet determines which strategy would have emitted each
signal.  No backtest re-run required.

Usage
─────
  uv run python scripts/eval_phase23_strategies.py
  uv run python scripts/eval_phase23_strategies.py --parquet data/signal_labels/my.parquet
  uv run python scripts/eval_phase23_strategies.py --folds 30 --fold-days 5
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Source → strategy mapping ─────────────────────────────────────────────────
_TREND_SOURCES  = {"technical_macd", "technical_ema_bounce"}
_RANGE_SOURCES  = {"technical_rsi", "technical_bollinger"}
_ALL_SOURCES    = _TREND_SOURCES | _RANGE_SOURCES

_TRENDING_REGIMES = {"BULL_TRENDING", "BEAR_TRENDING"}
_RANGING_REGIMES  = {"BULL_RANGING",  "BEAR_RANGING"}


def _in_strategy(row: pd.Series, strategy: str) -> bool:
    src    = row["source"]
    regime = str(row["regime"])
    if strategy == "v2_all":
        return src in _ALL_SOURCES
    if strategy == "trend_follower":
        return src in _TREND_SOURCES
    if strategy == "mean_reversion":
        return src in _RANGE_SOURCES
    if strategy == "routed_v23":
        if regime in _TRENDING_REGIMES:
            return src in _TREND_SOURCES
        if regime in _RANGING_REGIMES:
            return src in _RANGE_SOURCES
        return src in _ALL_SOURCES
    raise ValueError(strategy)


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    n_signals: int
    win_rate:  float
    mean_ret:  float
    sharpe:    float   # annualised daily Sharpe

    def to_row(self) -> dict:
        return {
            "n_signals": self.n_signals,
            "win_rate":  round(self.win_rate * 100, 1),
            "mean_ret":  round(self.mean_ret * 100, 3),
            "sharpe":    round(self.sharpe, 2),
        }


def _fold_metrics(fold_df: pd.DataFrame) -> FoldMetrics:
    if fold_df.empty:
        return FoldMetrics(0, 0.0, 0.0, 0.0)

    n        = len(fold_df)
    win_rate = float(fold_df["label"].mean())
    mean_ret = float(fold_df["net_ret"].mean())

    # daily P&L → annualised Sharpe
    daily = fold_df.set_index("bar_time")["net_ret"].resample("1D").sum()
    if len(daily) < 2 or daily.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(daily.mean() / daily.std() * math.sqrt(252))

    return FoldMetrics(n, win_rate, mean_ret, sharpe)


# ── Walk-forward ──────────────────────────────────────────────────────────────

STRATEGIES = ["v2_all", "trend_follower", "mean_reversion", "routed_v23"]


def run_wf(df: pd.DataFrame, n_folds: int, fold_days: int) -> dict[str, list[FoldMetrics]]:
    """Run walk-forward folds distributed evenly across the full date range."""
    df = df.sort_values("bar_time").copy()
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)

    t_start  = df["bar_time"].min()
    t_end    = df["bar_time"].max()
    total_td = (t_end - t_start).days
    # Stride so folds span the entire dataset evenly
    stride   = max(fold_days, (total_td - fold_days) // max(n_folds - 1, 1))

    results: dict[str, list[FoldMetrics]] = {s: [] for s in STRATEGIES}

    for i in range(n_folds):
        fold_start = t_start + pd.Timedelta(days=i * stride)
        fold_end   = fold_start + pd.Timedelta(days=fold_days)
        if fold_start >= t_end:
            break
        fold_df = df[(df["bar_time"] >= fold_start) & (df["bar_time"] < fold_end)]

        for strat in STRATEGIES:
            mask      = fold_df.apply(lambda r: _in_strategy(r, strat), axis=1)
            strat_df  = fold_df[mask]
            results[strat].append(_fold_metrics(strat_df))

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

@dataclass
class AggMetrics:
    mean_signals: float
    mean_win_rate: float
    mean_ret: float
    mean_sharpe: float
    std_sharpe: float
    positive_fold_pct: float


def _aggregate(folds: list[FoldMetrics]) -> AggMetrics:
    valid = [f for f in folds if f.n_signals > 0]
    if not valid:
        return AggMetrics(0, 0, 0, 0, 0, 0)
    sharpes = [f.sharpe for f in valid]
    return AggMetrics(
        mean_signals      = round(np.mean([f.n_signals for f in valid]), 1),
        mean_win_rate     = round(float(np.mean([f.win_rate for f in valid])) * 100, 1),
        mean_ret          = round(float(np.mean([f.mean_ret for f in valid])) * 100, 3),
        mean_sharpe       = round(float(np.mean(sharpes)), 2),
        std_sharpe        = round(float(np.std(sharpes)),  2),
        positive_fold_pct = round(sum(s > 0 for s in sharpes) / len(sharpes) * 100, 1),
    )


# ── Report ────────────────────────────────────────────────────────────────────

def _regime_breakdown(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Win-rate per regime per strategy (no walk-forward — full dataset)."""
    out: dict[str, dict[str, float]] = {}
    for strat in STRATEGIES:
        mask   = df.apply(lambda r: _in_strategy(r, strat), axis=1)
        sub    = df[mask]
        by_reg = sub.groupby("regime")["label"].agg(["mean", "count"])
        out[strat] = {
            reg: float(row["mean"]) for reg, row in by_reg.iterrows()
        }
    return out


def write_report(
    df: pd.DataFrame,
    results: dict[str, list[FoldMetrics]],
    n_folds: int,
    fold_days: int,
    out_path: Path,
) -> None:
    aggs = {s: _aggregate(results[s]) for s in STRATEGIES}
    regime_wr = _regime_breakdown(df)

    lines: list[str] = []
    lines += [
        "# Phase 2.3 — Multi-strategy routing: walk-forward comparison",
        "",
        f"Dataset : `{df['bar_time'].min().date()}` → `{df['bar_time'].max().date()}`  ",
        f"Rows    : {len(df):,}  |  Folds: {n_folds} × {fold_days}d",
        "",
        "## Summary — aggregated across all folds",
        "",
        "| Strategy | Signals/fold | Win rate | Mean ret | Sharpe (μ) | Sharpe (σ) | +folds |",
        "|----------|-------------|---------|---------|-----------|-----------|--------|",
    ]
    labels = {
        "v2_all":         "v2 (all det.)",
        "trend_follower": "trend_follower",
        "mean_reversion": "mean_reversion",
        "routed_v23":     "**routed_v23**",
    }
    for s in STRATEGIES:
        a = aggs[s]
        lines.append(
            f"| {labels[s]} | {a.mean_signals} | {a.mean_win_rate}% | "
            f"{a.mean_ret:+.3f}% | {a.mean_sharpe:.2f} | {a.std_sharpe:.2f} | "
            f"{a.positive_fold_pct}% |"
        )

    lines += [
        "",
        "## Win-rate by regime (full dataset)",
        "",
        "| Regime | v2_all | trend_follower | mean_reversion | routed_v23 |",
        "|--------|--------|----------------|----------------|------------|",
    ]
    all_regimes = sorted({r for strat_d in regime_wr.values() for r in strat_d})
    for reg in all_regimes:
        cells = [
            f"{regime_wr[s].get(reg, float('nan')) * 100:.1f}%"
            if not math.isnan(regime_wr[s].get(reg, float("nan")))
            else "—"
            for s in STRATEGIES
        ]
        lines.append(f"| {reg} | {' | '.join(cells)} |")

    lines += [
        "",
        "## Fold-by-fold Sharpe — routed_v23",
        "",
        "| Fold | Start | End | Signals | Win% | Sharpe |",
        "|------|-------|-----|---------|------|--------|",
    ]
    t_start  = df["bar_time"].min()
    t_end    = df["bar_time"].max()
    total_td = (t_end - t_start).days
    stride   = max(fold_days, (total_td - fold_days) // max(n_folds - 1, 1))
    for i, fm in enumerate(results["routed_v23"]):
        fs = (t_start + pd.Timedelta(days=i * stride)).date()
        fe = (t_start + pd.Timedelta(days=i * stride + fold_days)).date()
        lines.append(
            f"| {i+1} | {fs} | {fe} | {fm.n_signals} | "
            f"{fm.win_rate*100:.1f}% | {fm.sharpe:.2f} |"
        )

    lines += [
        "",
        "## Gate checks",
        "",
    ]
    rv23 = aggs["routed_v23"]
    base = aggs["v2_all"]
    checks = [
        ("routed_v23 Sharpe > v2_all",      rv23.mean_sharpe > base.mean_sharpe),
        ("routed_v23 win rate > 55%",        rv23.mean_win_rate > 55.0),
        ("routed_v23 positive folds > 60%",  rv23.positive_fold_pct > 60.0),
    ]
    for desc, passed in checks:
        icon = "✅" if passed else "❌"
        lines.append(f"- {icon} {desc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _latest_parquet() -> Path:
    d = Path("data/signal_labels")
    files = sorted(d.glob("labels_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No labeled parquet found in data/signal_labels/")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.3 strategy WF comparison")
    parser.add_argument("--parquet",   type=Path, default=None,
                        help="Labeled parquet (default: latest in data/signal_labels/)")
    parser.add_argument("--folds",     type=int, default=20)
    parser.add_argument("--fold-days", type=int, default=7)
    parser.add_argument("--out",       type=Path,
                        default=Path("docs/training_runs/phase2_3_strategy_comparison.md"))
    args = parser.parse_args()

    parquet = args.parquet or _latest_parquet()
    print(f"Loading {parquet} …")
    df = pd.read_parquet(parquet)
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    print(f"  {len(df):,} signals  |  {df['bar_time'].min().date()} → {df['bar_time'].max().date()}")

    print(f"Running {args.folds}-fold × {args.fold_days}d walk-forward …")
    results = run_wf(df, args.folds, args.fold_days)

    write_report(df, results, args.folds, args.fold_days, args.out)

    # print summary table to stdout
    print(f"\n{'='*68}")
    print(f"{'Strategy':<18} {'Signals':>8} {'WinRate':>8} {'Sharpe':>8} {'+Folds':>7}")
    print(f"{'-'*68}")
    for s in STRATEGIES:
        a = {
            "v2_all":         "v2 (all)",
            "trend_follower": "trend_follower",
            "mean_reversion": "mean_reversion",
            "routed_v23":     "routed_v23",
        }[s]
        ag = _aggregate(results[s])
        print(f"  {a:<16} {ag.mean_signals:>8.0f} {ag.mean_win_rate:>7.1f}% "
              f"{ag.mean_sharpe:>8.2f} {ag.positive_fold_pct:>6.0f}%")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
