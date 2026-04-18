"""T-L2-01-1: Verify conditional (regime-aware) signal lift over unconditional baseline.

Trains two LightGBM classifiers on 1h features with walk-forward cross-validation:
  A) Unconditional — no regime feature
  B) Conditional   — regime one-hot appended to feature matrix

Reports AUC per regime and overall for both models, then concludes whether
conditioning on L1 regime provides a statistically meaningful lift.

Usage::
    cd python
    uv run python scripts/verify_l2_lift.py \
        --symbol ETH/USDT --exchange binance \
        --forward-bars 4 --n-folds 3
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_REGIME_ORDER = [
    "bear_trending", "bear_ranging", "distribution",
    "accumulation", "bull_ranging", "bull_trending",
]
_MIN_DAILY = 60


async def _load_candles(symbol: str, exchange: str) -> dict[str, pd.DataFrame]:
    from smart_trader.core.env_files import load_repo_env_files
    load_repo_env_files()

    from sqlalchemy import text
    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    frames: dict[str, pd.DataFrame] = {}
    for tf in ("1h", "1d"):
        sql = text("""
            SELECT time, open::double precision, high::double precision,
                   low::double precision, close::double precision,
                   volume::double precision
            FROM candles
            WHERE symbol = :sym AND timeframe = :tf AND exchange = :ex
            ORDER BY time ASC
        """)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, {"sym": symbol, "tf": tf, "ex": exchange})).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df = df.set_index("time")
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            frames[tf] = df.sort_index()
    return frames


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


def _compute_regime_labels(daily_df: pd.DataFrame, primary_df: pd.DataFrame, symbol: str) -> pd.Series:
    from smart_trader.strategy.trend.engine import TrendEngine
    engine = TrendEngine()
    n = len(daily_df)
    labels: dict[pd.Timestamp, str] = {}
    for i in range(_MIN_DAILY, n, 6):
        ts = daily_df.index[i]
        try:
            state = engine.analyse(daily_df.iloc[: i + 1], symbol, "1d")
            labels[ts] = str(state.regime).lower()
        except Exception:
            labels[ts] = "unknown"

    s = pd.Series(labels, name="regime")
    s.index = pd.to_datetime(s.index, utc=True)
    combined = primary_df.index.union(s.index)
    s = s.reindex(combined).ffill().bfill()
    return s.reindex(primary_df.index).fillna("unknown")


def _build_dataset(
    df_1h: pd.DataFrame,
    regime_labels: pd.Series,
    forward_bars: int,
) -> pd.DataFrame:
    from smart_trader.data.features.engine import compute_features

    feats = compute_features(df_1h).dropna()
    fwd_ret = df_1h["close"].pct_change(forward_bars).shift(-forward_bars)
    target = (fwd_ret > 0).astype(int)

    combined = feats.copy()
    combined["_target"] = target
    combined["_regime"] = regime_labels.reindex(combined.index).ffill().fillna("unknown")
    combined = combined.dropna(subset=["_target"])
    return combined


def _train_eval(X_train, y_train, X_val, y_val) -> float:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    if len(np.unique(y_val)) < 2:
        return 0.5
    return roc_auc_score(y_val, proba)


def _walk_forward_auc(data: pd.DataFrame, feature_cols: list[str], n_folds: int) -> tuple[float, pd.Series]:
    """Returns (overall_auc, per-regime AUC series)."""
    fold_size = len(data) // (n_folds + 1)
    all_proba: list[float] = []
    all_target: list[int] = []
    all_regime: list[str] = []

    import lightgbm as lgb

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = val_start + fold_size

        train = data.iloc[:train_end]
        val = data.iloc[val_start:val_end]
        if len(train) < 200 or len(val) < 50:
            continue

        X_tr = train[feature_cols].values.astype(np.float32)
        y_tr = train["_target"].values
        X_val = val[feature_cols].values.astype(np.float32)
        y_val = val["_target"].values

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1].tolist()

        all_proba.extend(proba)
        all_target.extend(y_val.tolist())
        all_regime.extend(val["_regime"].tolist())

    all_proba_arr = np.array(all_proba)
    all_target_arr = np.array(all_target)
    all_regime_arr = np.array(all_regime)

    if len(np.unique(all_target_arr)) < 2:
        overall = 0.5
    else:
        overall = roc_auc_score(all_target_arr, all_proba_arr)

    per_regime: dict[str, float] = {}
    for reg in _REGIME_ORDER:
        mask = all_regime_arr == reg
        if mask.sum() < 20 or len(np.unique(all_target_arr[mask])) < 2:
            continue
        per_regime[reg] = roc_auc_score(all_target_arr[mask], all_proba_arr[mask])

    return overall, pd.Series(per_regime, name="auc")


def run(symbol: str, exchange: str, forward_bars: int, n_folds: int) -> None:
    print(f"Loading data for {symbol}…")
    frames = asyncio.run(_load_candles(symbol, exchange))

    df_1h = frames.get("1h")
    df_1d = frames.get("1d")
    if df_1h is None:
        print("No 1h data found."); return

    if df_1d is None or len(df_1d) < _MIN_DAILY:
        print("Resampling 1h → 1d…")
        df_1d = _resample_daily(df_1h)

    print(f"Computing regime labels ({len(df_1d)} daily bars)…")
    labels = _compute_regime_labels(df_1d, df_1h, symbol)

    print("Building feature dataset…")
    data = _build_dataset(df_1h, labels, forward_bars)

    feat_cols = [c for c in data.columns if c not in ("_target", "_regime")]

    # Regime one-hot columns
    regime_dummies = pd.get_dummies(data["_regime"], prefix="regime").astype(np.float32)
    data_cond = pd.concat([data, regime_dummies], axis=1)
    feat_cols_cond = feat_cols + list(regime_dummies.columns)

    print(f"\nWalk-forward {n_folds}-fold AUC (target: {forward_bars}-bar forward return sign)\n")

    print("Running Model A (unconditional)…")
    auc_a, per_regime_a = _walk_forward_auc(data, feat_cols, n_folds)

    print("Running Model B (conditional on regime)…")
    auc_b, per_regime_b = _walk_forward_auc(data_cond, feat_cols_cond, n_folds)

    lift = auc_b - auc_a

    print(f"\n{'═'*62}")
    print(f"  L2 Lift Verification — {symbol}  fwd={forward_bars}bar  folds={n_folds}")
    print(f"{'═'*62}")
    print(f"  {'Regime':<18} {'AUC-A (uncond)':>14} {'AUC-B (cond)':>13} {'lift':>7}")
    print(f"  {'─'*58}")

    all_regimes = sorted(set(list(per_regime_a.keys()) + list(per_regime_b.keys())))
    for reg in _REGIME_ORDER:
        if reg not in all_regimes:
            continue
        a = per_regime_a.get(reg, np.nan)
        b = per_regime_b.get(reg, np.nan)
        d = b - a if not (np.isnan(a) or np.isnan(b)) else np.nan
        flag = " ✓" if (not np.isnan(d) and d > 0.005) else ""
        print(f"  {reg:<18} {a:>14.4f} {b:>13.4f} {d:>+7.4f}{flag}")

    print(f"  {'─'*58}")
    print(f"  {'[overall]':<18} {auc_a:>14.4f} {auc_b:>13.4f} {lift:>+7.4f}")
    print(f"{'═'*62}")

    conclusion = "LIFT CONFIRMED" if lift > 0.005 else "NO SIGNIFICANT LIFT"
    print(f"\n  Conclusion: {conclusion}  (overall lift={lift:+.4f})")
    if lift > 0.005:
        print("  → L2 hypothesis holds: conditioning on regime improves signal quality.")
        print("  → Proceed to T-L2-01-2 / T-L3-01-1 (contextual RL).")
    else:
        print("  → L2 hypothesis not confirmed at this timeframe/horizon.")
        print("  → Document result and consider alternative conditioning strategies.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETH/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--forward-bars", type=int, default=4)
    ap.add_argument("--n-folds", type=int, default=3)
    args = ap.parse_args()
    run(args.symbol, args.exchange, args.forward_bars, args.n_folds)


if __name__ == "__main__":
    main()
