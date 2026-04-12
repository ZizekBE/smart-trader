"""Feature computation engine — vectorised indicator pipeline.

Computes all technical indicators from raw OHLCV in a single pass and
returns a feature DataFrame aligned by timestamp.  Supports multi-timeframe
feature stacking for the RL observation tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from smart_trader.strategy.trend.indicators import adx, ema, linear_slope, obv


@dataclass(slots=True, frozen=True)
class FeatureConfig:
    ema_periods: tuple[int, ...] = (9, 21, 50, 200)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    obv: bool = True
    slope_window: int = 14
    vwap: bool = True


def compute_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Compute all features from an OHLCV DataFrame.

    Args:
        df:     Must have columns [open, high, low, close, volume].
        config: Feature parameters.  Defaults to standard settings.
        prefix: Column name prefix (e.g. "1h_" for multi-TF stacking).

    Returns:
        DataFrame with feature columns, same index as input.
    """
    cfg = config or FeatureConfig()
    p = prefix
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feats: dict[str, pd.Series] = {}

    # --- price returns ---
    feats[f"{p}ret_1"] = c.pct_change(1)
    feats[f"{p}ret_5"] = c.pct_change(5)

    # --- EMAs (ratio only — no raw price-scale values) ---
    for period in cfg.ema_periods:
        e = ema(c, period)
        feats[f"{p}ema_{period}_dist"] = (c - e) / (e + 1e-9)

    # --- RSI (0-1 scaled) ---
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=cfg.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=cfg.rsi_period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    feats[f"{p}rsi"] = (100 - 100 / (1 + rs)) / 100.0

    # --- MACD (normalised by close) ---
    ema_fast = ema(c, cfg.macd_fast)
    ema_slow = ema(c, cfg.macd_slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, cfg.macd_signal)
    feats[f"{p}macd_pct"] = macd_line / (c + 1e-9)
    feats[f"{p}macd_signal_pct"] = signal_line / (c + 1e-9)
    feats[f"{p}macd_hist_pct"] = (macd_line - signal_line) / (c + 1e-9)

    # --- Bollinger Bands (normalised) ---
    bb_mid = c.rolling(cfg.bb_period).mean()
    bb_std = c.rolling(cfg.bb_period).std()
    feats[f"{p}bb_width"] = (2 * cfg.bb_std * bb_std) / (bb_mid + 1e-9)
    feats[f"{p}bb_pct"] = (c - (bb_mid - cfg.bb_std * bb_std)) / (2 * cfg.bb_std * bb_std + 1e-9)

    # --- ATR (ratio only) ---
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=cfg.atr_period, adjust=False).mean()
    feats[f"{p}atr_pct"] = atr_val / (c + 1e-9)

    # --- ADX (0-1 scaled) ---
    adx_val, di_plus, di_minus = adx(h, l, c, cfg.adx_period)
    feats[f"{p}adx"] = adx_val / 100.0
    feats[f"{p}di_plus"] = di_plus / 100.0
    feats[f"{p}di_minus"] = di_minus / 100.0

    # --- OBV (normalised slope) ---
    if cfg.obv:
        obv_val = obv(c, v)
        raw_slope = linear_slope(obv_val, cfg.slope_window)
        feats[f"{p}obv_slope_norm"] = raw_slope / (v.rolling(cfg.slope_window).mean() + 1e-9)

    # --- VWAP ---
    if cfg.vwap:
        tp = (h + l + c) / 3
        cum_tpv = (tp * v).cumsum()
        cum_v = v.cumsum()
        vwap_val = cum_tpv / (cum_v + 1e-9)
        feats[f"{p}vwap_dist"] = (c - vwap_val) / (vwap_val + 1e-9)

    # --- Volume features (ratio only) ---
    feats[f"{p}vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)

    # --- Trend slope (normalised by close) ---
    feats[f"{p}price_slope_pct"] = linear_slope(c, cfg.slope_window) / (c + 1e-9)

    return pd.DataFrame(feats, index=df.index)


def stack_multi_timeframe(
    ohlcv_by_tf: dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute features for each timeframe, align to target_index, and stack.

    Higher timeframes are forward-filled to the target resolution.
    """
    cfg = config or FeatureConfig()
    frames: list[pd.DataFrame] = []

    for tf, df in ohlcv_by_tf.items():
        feat = compute_features(df, cfg, prefix=f"{tf}_")
        feat = feat.reindex(target_index, method="ffill")
        frames.append(feat)

    return pd.concat(frames, axis=1)
