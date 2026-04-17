#!/usr/bin/env python3
"""Train a LightGBM binary classifier to filter rule-engine signals.

Label: 1 if T+20-bar net return > 0 (profitable after conservative round-trip fees).

Feature set (from SignalEvent.features + metadata):
  Numeric:  hist, hist_prev, vwap_factor, vol_ratio, vol_factor,
            vote_count, vote_boost, pct_b, rsi, rsi_prev,
            confidence, raw_score
  Encoded:  signal_type, source, regime, symbol  (LabelEncoded)

Raw price columns (close, upper, mid, ema20, low, high, lower, vwap) are
dropped — non-stationary, would leak scale info across time.

Train/val/test split is strictly time-ordered (no random shuffle).
  train : 2022-01-01 – 2025-06-30
  val   : 2025-07-01 – 2025-12-31
  test  : 2026-01-01 – present

Output:
  models/signal_filter/lgb_v1.model          (LightGBM booster)
  models/signal_filter/lgb_v1_encoders.pkl   (LabelEncoder mapping)
  models/signal_filter/lgb_v1_features.json  (ordered feature list)

Usage::

    uv run python scripts/train_signal_filter.py \\
        --data data/signal_labels/labels_ETHUSDT_BTCUSDT_h20_20260417_0232.parquet \\
        --output models/signal_filter
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
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

# Metadata / target / raw price columns to exclude from features
META_COLS = {"net_ret", "bar_time", "symbol", "signal_type", "source", "regime", "label"}
CAT_COLS  = ["signal_type", "source", "regime", "symbol"]

# Raw price detector columns — non-stationary, drop even with det_ prefix
_RAW_PRICE_SUFFIXES = {"close", "upper", "mid", "ema20", "ema50", "low", "high", "lower", "vwap"}

TRAIN_END = "2025-06-30"
VAL_END   = "2025-12-31"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM signal filter")
    p.add_argument("--data", required=True, help="Path to labeled .parquet file")
    p.add_argument("--output", default="models/signal_filter", help="Output directory")
    p.add_argument("--label-horizon", type=int, default=8, help="Forward bars used for labeling (must match collection)")
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    return p.parse_args()


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    df = df.sort_values("bar_time").reset_index(drop=True)
    log.info("loaded %d rows from %s", len(df), path)
    return df


def encode_cats(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    enc_out: dict = {} if encoders is None else encoders
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if encoders is None:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            enc_out[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels by mapping to -1
            mapping = {v: i for i, v in enumerate(le.classes_)}
            df[f"{col}_enc"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df, enc_out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded_cats = [f"{c}_enc" for c in CAT_COLS if c in df.columns]
    numeric = []
    for c in df.columns:
        if c in META_COLS or c in CAT_COLS or c.endswith("_enc") or c == "label":
            continue
        # Drop raw price detector columns
        suffix = c[4:] if c.startswith("det_") else None
        if suffix and suffix in _RAW_PRICE_SUFFIXES:
            continue
        numeric.append(c)
    feature_cols = numeric + encoded_cats
    return df[feature_cols].fillna(0.0), feature_cols


def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    # Precision@top_decile: of the top 10% by predicted probability, how many are positive?
    n_top = max(1, len(y) // 10)
    top_idx = np.argsort(proba)[-n_top:]
    prec_top = y.iloc[top_idx].mean()

    log.info(
        "%s  n=%d  AUC=%.4f  PR-AUC=%.4f  Prec@top10%%=%.4f  label_rate=%.3f",
        name, len(y), auc, pr_auc, prec_top, y.mean(),
    )
    return {"n": len(y), "auc": auc, "pr_auc": pr_auc, "prec_top_decile": prec_top,
            "label_rate": float(y.mean())}


def main() -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed — run: uv add lightgbm")
        sys.exit(1)
    try:
        from sklearn.metrics import roc_auc_score  # noqa: F401
    except ImportError:
        log.error("scikit-learn not installed — run: uv add scikit-learn")
        sys.exit(1)

    args = parse_args()
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.data)
    df, encoders = encode_cats(df)
    X_all, feature_cols = build_feature_matrix(df)
    y_all = df["label"]
    t = df["bar_time"]

    # Time-ordered split
    mask_train = t <= pd.Timestamp(TRAIN_END, tz="UTC")
    mask_val   = (t > pd.Timestamp(TRAIN_END, tz="UTC")) & (t <= pd.Timestamp(VAL_END, tz="UTC"))
    mask_test  = t > pd.Timestamp(VAL_END, tz="UTC")

    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_val,   y_val   = X_all[mask_val],   y_all[mask_val]
    X_test,  y_test  = X_all[mask_test],  y_all[mask_test]

    log.info("split  train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    # Train
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        verbose=-1,
    )

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    log.info("best iteration: %d", model.best_iteration_)

    # Evaluate
    results: dict = {
        "train": evaluate("train", model, X_train, y_train),
        "val":   evaluate("val",   model, X_val,   y_val),
        "test":  evaluate("test",  model, X_test,  y_test) if mask_test.any() else {},
    }

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi_path = out_dir / "lgb_v1_feature_importance.csv"
    fi.to_csv(fi_path, header=["importance"])
    log.info("top features:\n%s", fi.head(15).to_string())

    # Save artifacts
    model_path = out_dir / "lgb_v1.model"
    model.booster_.save_model(str(model_path))
    log.info("model saved → %s", model_path)

    enc_path = out_dir / "lgb_v1_encoders.pkl"
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    log.info("encoders saved → %s", enc_path)

    feat_path = out_dir / "lgb_v1_features.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info("feature list saved → %s", feat_path)

    # Summary
    results["config"] = {
        "data": args.data,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "n_estimators": args.n_estimators,
        "best_iteration": int(model.best_iteration_),
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "features": feature_cols,
    }
    summary_path = out_dir / "lgb_v1_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("results saved → %s", summary_path)

    # Gate check
    val_auc = results["val"]["auc"]
    val_prec = results["val"]["prec_top_decile"]
    log.info("--- GATE CHECK ---")
    log.info("val AUC=%.4f (gate: ≥0.55)  %s", val_auc, "✓ PASS" if val_auc >= 0.55 else "✗ FAIL")
    log.info("val Prec@top10%%=%.4f (gate: ≥0.60)  %s", val_prec, "✓ PASS" if val_prec >= 0.60 else "✗ FAIL")
    if val_auc >= 0.55 and val_prec >= 0.60:
        log.info("GATE PASSED — eligible for shadow integration")
    else:
        log.warning("GATE FAILED — do not integrate into main loop")


if __name__ == "__main__":
    main()
