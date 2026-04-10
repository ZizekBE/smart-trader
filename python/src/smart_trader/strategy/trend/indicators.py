"""Pure-numpy/pandas technical indicator calculations.

No TA-Lib dependency — all indicators implemented from first principles.
Performance: linear_slope and obv use vectorised numpy operations to avoid
the ~100× overhead of pandas rolling.apply(python_function).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX.  Returns (ADX, +DI, -DI)."""
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    raw_plus  =  high.diff()
    raw_minus = -low.diff()

    # keep only the dominant move in each bar
    dm_plus  = raw_plus.where((raw_plus > raw_minus) & (raw_plus > 0), 0.0)
    dm_minus = raw_minus.where((raw_minus >= raw_plus) & (raw_minus > 0), 0.0)

    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr + 1e-9)

    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx_line = dx.ewm(span=period, adjust=False).mean()

    return adx_line, di_plus, di_minus


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — vectorised via numpy sign."""
    diff = np.diff(close.to_numpy(), prepend=close.iloc[0])
    sign = np.sign(diff)
    return pd.Series((sign * volume.to_numpy()).cumsum(), index=close.index)


def linear_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope, normalised by the window mean.

    Returns a *relative* slope so it is comparable across instruments
    and timeframes.

    Vectorised implementation using sliding_window_view: ~50× faster than
    the previous rolling.apply(python_function) approach.
    """
    arr = series.to_numpy(dtype=float)
    n   = len(arr)

    if n < window:
        return pd.Series(np.full(n, np.nan), index=series.index)

    # Pre-compute fixed OLS x weights for this window size.
    # For x = [0, 1, ..., w-1], x_mean = (w-1)/2.
    # Σ(x_i - x_mean)^2 is a constant; the OLS slope reduces to a
    # simple dot product with fixed weights w_i = (x_i - x_mean) / den.
    x      = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_c    = x - x_mean
    den    = (x_c ** 2).sum()          # scalar constant

    # Build (n - window + 1, window) view with zero-copy stride tricks.
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)  # shape: (n-w+1, w)

    y_means  = windows.mean(axis=1)                  # (n-w+1,)
    y_c      = windows - y_means[:, None]            # (n-w+1, w)  — demeaned
    slopes   = (y_c @ x_c) / den                    # (n-w+1,)  — unnormalised
    rel      = np.where(np.abs(y_means) > 1e-9,
                        slopes / y_means, 0.0)       # normalise by mean

    result = np.full(n, np.nan)
    result[window - 1:] = rel
    return pd.Series(result, index=series.index)
