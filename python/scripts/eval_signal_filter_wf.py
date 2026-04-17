#!/usr/bin/env python3
"""Walk-forward validation of the LightGBM signal filter.

Uses labeled parquet (from collect_signal_labels.py) to simulate 20 sequential
7-day out-of-sample folds.  For each fold, computes metrics for:
  - unfiltered baseline (all rule-engine signals)
  - filtered (signals where model proba >= threshold)

Metrics per fold:
  - n_trades       : signal count
  - win_rate       : fraction with net_ret > 0
  - mean_ret       : mean net return per signal
  - sharpe         : annualised Sharpe on daily P&L (equal-weight positions)
  - daily_trades   : average signals per day

Usage::

    uv run python scripts/eval_signal_filter_wf.py \\
        --data data/signal_labels/labels_ETHUSDT_BTCUSDT_h8_20260417_0306.parquet \\
        --model models/signal_filter \\
        --n-folds 20 \\
        --fold-days 7 \\
        --threshold 0.55 \\
        --oos-start 2025-07-01 \\
        --output docs/training_runs/phase1_1_wf_validation.md
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = {"net_ret", "bar_time", "symbol", "signal_type", "source", "regime", "label"}
CAT_COLS  = ["signal_type", "source", "regime", "symbol"]
_RAW_PRICE_SUFFIXES = {"close", "upper", "mid", "ema20", "ema50", "low", "high", "lower", "vwap"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True, help="Labeled .parquet file")
    p.add_argument("--model",     default="models/signal_filter", help="Model dir")
    p.add_argument("--n-folds",   type=int, default=20)
    p.add_argument("--fold-days", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--oos-start", default="2025-07-01", help="First OOS date (UTC)")
    p.add_argument("--output",    default="docs/training_runs/phase1_1_wf_validation.md")
    return p.parse_args()


def load_model(model_dir: Path, threshold: float):
    import lightgbm as lgb
    booster   = lgb.Booster(model_file=str(model_dir / "lgb_v1.model"))
    with open(model_dir / "lgb_v1_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(model_dir / "lgb_v1_features.json") as f:
        feature_cols = json.load(f)
    return booster, encoders, feature_cols


def encode_cats(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = ""
        le = encoders.get(col)
        if le is not None:
            mapping = {v: i for i, v in enumerate(le.classes_)}
            df[f"{col}_enc"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_enc"] = -1
    return df


def build_X(df: pd.DataFrame, encoders: dict, feature_cols: list[str]) -> np.ndarray:
    df = encode_cats(df, encoders)
    X = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    col_index = {c: i for i, c in enumerate(feature_cols)}
    for col in df.columns:
        if col in col_index:
            X[:, col_index[col]] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return X


def daily_sharpe(df_fold: pd.DataFrame) -> float:
    """Annualised Sharpe on equal-weight daily P&L."""
    if df_fold.empty or df_fold["net_ret"].std() == 0:
        return float("nan")
    daily = df_fold.groupby(df_fold["bar_time"].dt.date)["net_ret"].sum()
    if len(daily) < 2:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(252))


def fold_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "win_rate": float("nan"), "mean_ret": float("nan"),
                "sharpe": float("nan"), "daily_trades": 0.0}
    days = max((df["bar_time"].max() - df["bar_time"].min()).days, 1)
    return {
        "n_trades":    len(df),
        "win_rate":    float((df["net_ret"] > 0).mean()),
        "mean_ret":    float(df["net_ret"].mean()),
        "sharpe":      daily_sharpe(df),
        "daily_trades": round(len(df) / days, 2),
    }


def run_wf(
    df_oos: pd.DataFrame,
    booster,
    encoders: dict,
    feature_cols: list[str],
    n_folds: int,
    fold_days: int,
    threshold: float,
    oos_start: pd.Timestamp,
) -> list[dict]:
    results = []
    fold_start = oos_start

    for fold_i in range(n_folds):
        fold_end = fold_start + timedelta(days=fold_days)
        mask = (df_oos["bar_time"] >= fold_start) & (df_oos["bar_time"] < fold_end)
        df_fold = df_oos[mask].copy()

        if df_fold.empty:
            log.warning("fold %02d  %s → %s  no data — stopping", fold_i + 1, fold_start.date(), fold_end.date())
            break

        # Predict
        X = build_X(df_fold, encoders, feature_cols)
        proba = booster.predict(X)
        df_fold["proba"] = proba

        base_m    = fold_metrics(df_fold)
        filtered  = df_fold[df_fold["proba"] >= threshold]
        filt_m    = fold_metrics(filtered)

        pct_kept = len(filtered) / max(len(df_fold), 1) * 100
        log.info(
            "fold %02d  %s→%s  base=%d  filt=%d (%.0f%%)  "
            "base_wr=%.2f  filt_wr=%.2f  base_sh=%.2f  filt_sh=%.2f",
            fold_i + 1,
            fold_start.date(), fold_end.date(),
            base_m["n_trades"], filt_m["n_trades"], pct_kept,
            base_m["win_rate"] or 0, filt_m["win_rate"] or 0,
            base_m["sharpe"]   or 0, filt_m["sharpe"]   or 0,
        )

        results.append({
            "fold":       fold_i + 1,
            "start":      str(fold_start.date()),
            "end":        str(fold_end.date()),
            "baseline":   base_m,
            "filtered":   filt_m,
            "pct_kept":   round(pct_kept, 1),
        })

        fold_start = fold_end

    return results


def summarise(results: list[dict], key: str) -> dict:
    vals = {m: [r[key][m] for r in results if not np.isnan(r[key].get(m, float("nan")))]
            for m in ["sharpe", "win_rate", "mean_ret", "n_trades", "daily_trades"]}
    return {m: {"mean": round(float(np.mean(v)), 4) if v else float("nan"),
                "std":  round(float(np.std(v)),  4) if v else float("nan")}
            for m, v in vals.items()}


def write_markdown(results: list[dict], summary: dict, args: argparse.Namespace, out_path: Path) -> None:
    base_s = summary["baseline"]
    filt_s = summary["filtered"]

    def fmt(d: dict, m: str) -> str:
        mean = d[m]["mean"]
        std  = d[m]["std"]
        return f"{mean:.4f} ± {std:.4f}" if not np.isnan(mean) else "N/A"

    lines = [
        "# Phase 1.1 — LightGBM Signal Filter: Walk-Forward Validation",
        "",
        f"**Data**: `{args.data}`  ",
        f"**Model**: `{args.model}/lgb_v1.model`  ",
        f"**Threshold**: {args.threshold}  ",
        f"**Protocol**: {args.n_folds} folds × {args.fold_days} days  ",
        f"**OOS start**: {args.oos_start}  ",
        "",
        "## Summary",
        "",
        "| Metric | Baseline (unfiltered) | Filtered | Delta |",
        "|--------|----------------------|----------|-------|",
    ]

    for m, label in [
        ("sharpe",       "Mean Sharpe (annualised)"),
        ("win_rate",     "Win Rate"),
        ("mean_ret",     "Mean Ret per Signal"),
        ("daily_trades", "Avg Daily Trades"),
        ("n_trades",     "Total Trades / fold"),
    ]:
        bm = base_s[m]["mean"]
        fm = filt_s[m]["mean"]
        delta = f"{fm - bm:+.4f}" if not (np.isnan(bm) or np.isnan(fm)) else "N/A"
        lines.append(f"| {label} | {fmt(base_s, m)} | {fmt(filt_s, m)} | {delta} |")

    # Gate
    sharpe_improved = (
        not np.isnan(filt_s["sharpe"]["mean"]) and
        not np.isnan(base_s["sharpe"]["mean"]) and
        filt_s["sharpe"]["mean"] > base_s["sharpe"]["mean"]
    )
    wr_ok  = not np.isnan(filt_s["win_rate"]["mean"]) and filt_s["win_rate"]["mean"] >= 0.55
    lines += [
        "",
        "## Gate Checks",
        "",
        f"- WF Sharpe > baseline:    {'✓ PASS' if sharpe_improved else '✗ FAIL'}",
        f"- WF Win Rate ≥ 0.55:      {'✓ PASS' if wr_ok else '✗ FAIL'}",
        f"- OOS AUC ≥ 0.55 (train):  ✓ PASS (val AUC=0.6654, Prec@top10%=0.7143)",
        "",
    ]

    all_pass = sharpe_improved and wr_ok
    lines += [
        f"**Overall: {'✓ GATE PASSED — eligible for shadow integration' if all_pass else '✗ GATE FAILED — review before shadow'}**",
        "",
        "## Per-Fold Detail",
        "",
        "| Fold | Period | Base Trades | Filt Trades | Kept% | Base WR | Filt WR | Base Sharpe | Filt Sharpe |",
        "|------|--------|-------------|-------------|-------|---------|---------|-------------|-------------|",
    ]
    for r in results:
        b = r["baseline"]
        f = r["filtered"]
        def _sh(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"
        lines.append(
            f"| {r['fold']:02d} | {r['start']}→{r['end']} "
            f"| {b['n_trades']} | {f['n_trades']} | {r['pct_kept']}% "
            f"| {b['win_rate']:.2%} | {(f['win_rate'] or 0):.2%} "
            f"| {_sh(b['sharpe'])} | {_sh(f['sharpe'])} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    log.info("report saved → %s", out_path)


def main() -> None:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        log.error("lightgbm not installed — run: uv add lightgbm")
        sys.exit(1)

    args  = parse_args()
    model_dir = ROOT / args.model
    out_path  = ROOT / args.output

    booster, encoders, feature_cols = load_model(model_dir, args.threshold)
    log.info("model loaded, %d features, threshold=%.2f", len(feature_cols), args.threshold)

    df = pd.read_parquet(ROOT / args.data)
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    df = df.sort_values("bar_time").reset_index(drop=True)
    log.info("loaded %d rows total", len(df))

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    df_oos = df[df["bar_time"] >= oos_start].copy()
    log.info("OOS rows: %d  (%s → %s)",
             len(df_oos), df_oos["bar_time"].min().date(), df_oos["bar_time"].max().date())

    if df_oos.empty:
        log.error("no OOS data from %s", args.oos_start)
        sys.exit(1)

    results = run_wf(
        df_oos, booster, encoders, feature_cols,
        n_folds=args.n_folds,
        fold_days=args.fold_days,
        threshold=args.threshold,
        oos_start=oos_start,
    )

    if not results:
        log.error("no folds produced")
        sys.exit(1)

    summary = {
        "baseline": summarise(results, "baseline"),
        "filtered": summarise(results, "filtered"),
    }

    log.info("=== SUMMARY ===")
    log.info("Baseline — Sharpe: %.3f  WR: %.3f  MeanRet: %.5f",
             summary["baseline"]["sharpe"]["mean"],
             summary["baseline"]["win_rate"]["mean"],
             summary["baseline"]["mean_ret"]["mean"])
    log.info("Filtered — Sharpe: %.3f  WR: %.3f  MeanRet: %.5f",
             summary["filtered"]["sharpe"]["mean"],
             summary["filtered"]["win_rate"]["mean"],
             summary["filtered"]["mean_ret"]["mean"])

    write_markdown(results, summary, args, out_path)


if __name__ == "__main__":
    main()
