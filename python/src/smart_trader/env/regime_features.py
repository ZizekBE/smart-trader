"""Batch pre-computation of regime/volatility features for RL training.

Produces a DataFrame indexed by primary-timeframe timestamps with two
normalised float columns:
  regime_code   — MarketRegime ordinal, normalised to [0, 1]
  vol_code      — VolRegime ordinal, normalised to [0, 1]

These are appended to the RL agent's context vector (SpaceConfig.regime_dim=2)
so the agent can condition its policy on the current market regime without
re-running TrendEngine/VolatilityAnalyzer on every env step.

Usage::
    from smart_trader.env.regime_features import compute_regime_features
    reg_df = compute_regime_features(daily_df, hourly_df, primary_df, "ETH/USDT")
    # reg_df has columns ['regime_code', 'vol_code'], same index as primary_df
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Ordinal encoding: more directional / stronger trend → higher value
_REGIME_ORDER: dict[str, float] = {
    "bear_trending":  0.0,
    "bear_ranging":   0.2,
    "distribution":   0.35,
    "accumulation":   0.65,
    "bull_ranging":   0.8,
    "bull_trending":  1.0,
}

_VOL_ORDER: dict[str, float] = {
    "low":    0.0,
    "medium": 0.33,
    "high":   0.67,
    "crisis": 1.0,
}

_DAILY_MIN_BARS = 60   # TrendEngine minimum
_HOURLY_MIN_BARS = 24  # VolatilityAnalyzer minimum


def _encode_regime(regime_str: str) -> float:
    return _REGIME_ORDER.get(str(regime_str).lower(), 0.5)


def _encode_vol(vol_str: str) -> float:
    return _VOL_ORDER.get(str(vol_str).lower(), 0.33)


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample OHLCV DataFrame to daily bars (OHLCV aggregation)."""
    df = _ensure_dt_index(df)
    daily = df.resample("1D").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return daily


def compute_regime_features(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    symbol: str,
    stride: int = 1,
) -> pd.DataFrame:
    """Compute per-bar regime + vol features aligned to primary_df index.

    Args:
        daily_df:   Daily OHLCV (sorted ascending).
        hourly_df:  Hourly OHLCV (sorted ascending).
        primary_df: Primary-timeframe OHLCV (target index).
        symbol:     Symbol string for TrendEngine.
        stride:     Compute every N daily bars (1 = every bar, 6 = weekly).
                    Intermediate bars are forward-filled.

    Returns:
        DataFrame with columns ['regime_code', 'vol_code'] indexed by
        primary_df.index.  All values ∈ [0, 1].
    """
    from smart_trader.strategy.trend.engine import TrendEngine
    from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer

    trend_engine = TrendEngine()
    vol_analyzer = VolatilityAnalyzer()

    # ensure datetime index
    daily_df   = _ensure_dt_index(daily_df)
    hourly_df  = _ensure_dt_index(hourly_df)
    primary_df = _ensure_dt_index(primary_df)

    # fall back to resampling hourly if daily is missing or too short
    if daily_df.empty or len(daily_df) < _DAILY_MIN_BARS:
        if not hourly_df.empty:
            log.info("regime_features: resampling 1h→1d for %s", symbol)
            daily_df = _resample_to_daily(hourly_df)
        if len(daily_df) < _DAILY_MIN_BARS:
            log.warning("regime_features: not enough daily bars for %s after resample (%d)", symbol, len(daily_df))
            return pd.DataFrame({"regime_code": 0.5, "vol_code": 0.33}, index=primary_df.index)

    regime_series: dict[pd.Timestamp, float] = {}
    vol_series:    dict[pd.Timestamp, float] = {}

    n_daily = len(daily_df)
    computed = 0

    for i in range(_DAILY_MIN_BARS, n_daily):
        if (i - _DAILY_MIN_BARS) % stride != 0 and i != n_daily - 1:
            continue
        ts = daily_df.index[i]

        # regime
        try:
            state = trend_engine.analyse(daily_df.iloc[: i + 1], symbol, "1d")
            regime_series[ts] = _encode_regime(str(state.regime))
        except Exception:
            regime_series[ts] = 0.5

        # vol — use daily bars up to ts + hourly bars up to ts
        h_slice = hourly_df[hourly_df.index <= ts]
        if len(h_slice) >= _HOURLY_MIN_BARS:
            try:
                vs = vol_analyzer.analyse(daily_df.iloc[: i + 1], h_slice)
                vol_series[ts] = _encode_vol(str(vs.regime))
            except Exception:
                vol_series[ts] = 0.33
        else:
            vol_series[ts] = 0.33

        computed += 1

    if not regime_series:
        log.warning("regime_features: no data computed for %s — returning zeros", symbol)
        return pd.DataFrame(
            {"regime_code": 0.5, "vol_code": 0.33},
            index=primary_df.index,
        )

    reg_s = pd.Series(regime_series, name="regime_code")
    vol_s = pd.Series(vol_series,    name="vol_code")

    reg_df = pd.DataFrame({"regime_code": reg_s, "vol_code": vol_s})
    reg_df.index = pd.to_datetime(reg_df.index, utc=True)

    # reindex to primary timeframe and forward-fill
    combined_idx = primary_df.index.union(reg_df.index)
    reg_df = reg_df.reindex(combined_idx).ffill().bfill()
    reg_df = reg_df.reindex(primary_df.index)
    reg_df = reg_df.fillna({"regime_code": 0.5, "vol_code": 0.33})

    log.info(
        "regime_features computed  symbol=%s  daily_bars=%d  computed_points=%d  coverage=%.1f%%",
        symbol, n_daily, computed,
        100 * reg_df["regime_code"].notna().mean(),
    )
    return reg_df.astype(np.float32)


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()
