"""LightGBM signal filter — post-filters rule-engine SignalEvents.

Usage::

    filter = SignalFilter.from_model_dir("models/signal_filter")
    kept = filter.filter(events, df_1h=df_1h, df_4h=df_4h)

The filter accepts events from SignalEngine.analyse(), computes the same feature
vector used during training (1h context + 4h context + detector features + time),
and drops signals whose estimated win-probability is below `threshold`.

Model artefacts expected in model_dir:
  lgb_v1.model          — LightGBM booster (text format)
  lgb_v1_encoders.pkl   — dict[str, LabelEncoder]
  lgb_v1_features.json  — ordered feature column list
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from smart_trader.strategy.signals.models import SignalEvent

log = structlog.get_logger(__name__)

_CAT_COLS   = ["signal_type", "source", "regime", "symbol"]
_RAW_PRICE  = {"close", "upper", "mid", "ema20", "ema50", "low", "high", "lower", "vwap"}
_META_COLS  = {"net_ret", "bar_time", "symbol", "signal_type", "source", "regime", "label"}


class SignalFilter:
    def __init__(
        self,
        booster,
        encoders: dict,
        feature_cols: list[str],
        threshold: float = 0.55,
    ) -> None:
        self._booster      = booster
        self._encoders     = encoders
        self._feature_cols = feature_cols
        self.threshold     = threshold

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, threshold: float = 0.55) -> "SignalFilter":
        import lightgbm as lgb

        d = Path(model_dir)
        booster = lgb.Booster(model_file=str(d / "lgb_v1.model"))
        with open(d / "lgb_v1_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(d / "lgb_v1_features.json") as f:
            feature_cols = json.load(f)
        log.info("signal_filter_loaded", model_dir=str(d), features=len(feature_cols), threshold=threshold)
        return cls(booster, encoders, feature_cols, threshold)

    def filter(
        self,
        events: list["SignalEvent"],
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        symbol: str = "",
    ) -> list["SignalEvent"]:
        if not events:
            return events

        from smart_trader.data.features.engine import compute_features

        # Compute normalized feature matrices (DatetimeIndex input)
        feat_1h = compute_features(df_1h) if not df_1h.empty else pd.DataFrame()
        feat_4h = compute_features(df_4h, prefix="4h_") if not df_4h.empty else pd.DataFrame()

        rows: list[dict] = []
        for ev in events:
            bar_time = ev.created_at

            # 1h context features (last bar at or before signal time)
            ctx: dict = {}
            if not feat_1h.empty:
                idx_1h = feat_1h.index.searchsorted(bar_time, side="right") - 1
                if idx_1h >= 0:
                    ctx.update({f"ctx_{k}": float(v) for k, v in feat_1h.iloc[idx_1h].items()
                                if not pd.isna(v)})

            # 4h context features (asof)
            if not feat_4h.empty:
                idx_4h = feat_4h.index.searchsorted(bar_time, side="right") - 1
                if idx_4h >= 0:
                    ctx.update({k: float(v) for k, v in feat_4h.iloc[idx_4h].items()
                                if not pd.isna(v)})

            # Detector features (prefixed)
            det = {f"det_{k}": v for k, v in ev.features.items()
                   if f"det_{k[4:]}" not in _RAW_PRICE}  # skip raw prices

            row: dict = {
                "signal_type": ev.signal_type,
                "source":      ev.source,
                "regime":      ev.regime.name if hasattr(ev.regime, "name") else str(ev.regime),
                "symbol":      symbol or ev.symbol,
                "confidence":  ev.confidence,
                "raw_score":   ev.raw_score,
                "hour_of_day": bar_time.hour,
                "day_of_week": bar_time.weekday(),
            }
            row.update(det)
            row.update(ctx)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Encode categoricals using saved LabelEncoders
        for col in _CAT_COLS:
            if col not in df.columns:
                df[col] = ""
            le = self._encoders.get(col)
            if le is not None:
                mapping = {v: i for i, v in enumerate(le.classes_)}
                df[f"{col}_enc"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
            else:
                df[f"{col}_enc"] = -1

        # Build feature matrix aligned to training feature order
        X = np.zeros((len(df), len(self._feature_cols)), dtype=np.float32)
        col_index = {c: i for i, c in enumerate(self._feature_cols)}
        for col in df.columns:
            if col in col_index:
                X[:, col_index[col]] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

        proba = self._booster.predict(X)

        kept = [ev for ev, p in zip(events, proba) if p >= self.threshold]
        dropped = len(events) - len(kept)
        if dropped:
            log.info("signal_filter_applied", kept=len(kept), dropped=dropped,
                     threshold=self.threshold)
        return kept
