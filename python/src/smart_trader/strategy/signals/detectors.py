"""Individual technical signal detectors.

Each detector function accepts a pandas DataFrame with columns
(close, high, low, volume) and returns a DetectorResult or None.

DetectorResult = (signal_type, raw_score, features)
  signal_type : 'long' | 'short'
  raw_score   : float in [0, 1]  — detector strength before context scoring
  features    : dict of indicator snapshot for auditing
"""
from __future__ import annotations

import math
from typing import TypeAlias

import pandas as pd

# ── type alias ────────────────────────────────────────────────
DetectorResult: TypeAlias = tuple[str, float, dict] | None


# ═══════════════════════════════════════════════════════════════
#  Indicator helpers (private, minimal, no TA-Lib)
# ═══════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower, %B)."""
    mid     = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper   = mid + n_std * std_dev
    lower   = mid - n_std * std_dev
    pct_b   = (close - lower) / (upper - lower + 1e-9)
    return upper, mid, lower, pct_b


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════
#  Detectors
# ═══════════════════════════════════════════════════════════════

def detect_rsi(df: pd.DataFrame) -> DetectorResult:
    """RSI oversold / overbought detector.

    Long  when RSI < 30 (deeper & turning up = stronger).
    Short when RSI > 70 (higher & turning down = stronger).
    """
    if len(df) < 15:
        return None
    close  = df["close"].astype(float)
    series = _rsi(close)
    rsi    = series.iloc[-1]
    prev   = series.iloc[-2]

    if math.isnan(rsi):
        return None

    if rsi < 30:
        depth = min(1.0, (30 - rsi) / 25)          # 0→1 as rsi: 30→5
        turn  = 1.15 if rsi > prev else 1.0         # bonus if reversing
        score = min(1.0, depth * turn)
        return "long", score, {"rsi": rsi, "rsi_prev": prev}

    if rsi > 70:
        depth = min(1.0, (rsi - 70) / 25)
        turn  = 1.15 if rsi < prev else 1.0
        score = min(1.0, depth * turn)
        return "short", score, {"rsi": rsi, "rsi_prev": prev}

    return None


def detect_macd_cross(df: pd.DataFrame) -> DetectorResult:
    """MACD histogram zero-line cross detector.

    Fires on the first bar the histogram flips sign (momentum shift).
    Score is proportional to histogram magnitude relative to price.
    """
    if len(df) < 28:
        return None
    close = df["close"].astype(float)
    _, _, hist = _macd(close)

    curr = hist.iloc[-1]
    prev = hist.iloc[-2]

    if math.isnan(curr) or math.isnan(prev):
        return None

    if prev <= 0 < curr:          # bullish cross
        magnitude = abs(curr) / (close.iloc[-1] * 0.002 + 1e-9)
        score = min(1.0, magnitude)
        return "long",  score, {"hist": curr, "hist_prev": prev}

    if prev >= 0 > curr:          # bearish cross
        magnitude = abs(curr) / (close.iloc[-1] * 0.002 + 1e-9)
        score = min(1.0, magnitude)
        return "short", score, {"hist": curr, "hist_prev": prev}

    return None


def detect_bollinger_touch(df: pd.DataFrame) -> DetectorResult:
    """Bollinger Band touch / squeeze detector.

    %B < 0.05 → price near lower band → long candidate.
    %B > 0.95 → price near upper band → short candidate.
    Score from how far outside the band the price reached.
    """
    if len(df) < 21:
        return None
    close = df["close"].astype(float)
    upper, mid, lower, pct_b_series = _bollinger(close)
    pb = pct_b_series.iloc[-1]

    if math.isnan(pb):
        return None

    if pb < 0.05:
        score = min(1.0, (0.1 - pb) / 0.1)
        return "long", score, {
            "pct_b": pb, "close": close.iloc[-1],
            "lower": lower.iloc[-1], "mid": mid.iloc[-1],
        }
    if pb > 0.95:
        score = min(1.0, (pb - 0.9) / 0.1)
        return "short", score, {
            "pct_b": pb, "close": close.iloc[-1],
            "upper": upper.iloc[-1], "mid": mid.iloc[-1],
        }
    return None


def detect_ema_bounce(df: pd.DataFrame) -> DetectorResult:
    """EMA bounce detector — price pulls back to key EMA in a trend.

    Long:  low touches EMA20 but close holds above it (uptrend pullback).
    Short: high touches EMA20 but close stays below it (downtrend bounce).
    Score based on proximity: closer to EMA = stronger setup.
    """
    if len(df) < 21:
        return None
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)

    e20 = _ema(close, 20)
    e50 = _ema(close, 50) if len(df) >= 50 else None

    ema_val  = e20.iloc[-1]
    c        = close.iloc[-1]
    h        = high.iloc[-1]
    l        = low.iloc[-1]
    bar_size = h - l + 1e-9

    # long bounce: low dipped to/below EMA but closed above it
    if l <= ema_val <= c:
        proximity = 1 - (c - ema_val) / (h - ema_val + 1e-9)
        score = max(0.0, min(1.0, proximity * 0.8))
        if score < 0.2:
            return None
        features = {"ema20": ema_val, "close": c, "low": l}
        if e50 is not None:
            features["ema50"] = e50.iloc[-1]
        return "long", score, features

    # short bounce: high reached/above EMA but closed below it
    if c <= ema_val <= h:
        proximity = 1 - (ema_val - c) / (ema_val - l + 1e-9)
        score = max(0.0, min(1.0, proximity * 0.8))
        if score < 0.2:
            return None
        features = {"ema20": ema_val, "close": c, "high": h}
        if e50 is not None:
            features["ema50"] = e50.iloc[-1]
        return "short", score, features

    return None


def detect_ema_crossover(df: pd.DataFrame) -> DetectorResult:
    """EMA 9/21 crossover detector — fires on the bar a crossover occurs.

    Long:  EMA9 crosses above EMA21 (bullish momentum shift).
    Short: EMA9 crosses below EMA21 (bearish momentum shift).
    Score is proportional to the gap size relative to recent ATR, so
    wider crosses score higher than marginal ones.
    Requires at least 30 bars.
    """
    if len(df) < 30:
        return None
    close = df["close"].astype(float)
    e9    = _ema(close, 9)
    e21   = _ema(close, 21)

    curr_diff = e9.iloc[-1]  - e21.iloc[-1]
    prev_diff = e9.iloc[-2]  - e21.iloc[-2]

    if math.isnan(curr_diff) or math.isnan(prev_diff):
        return None

    # no crossover
    if (curr_diff > 0) == (prev_diff > 0):
        return None

    # ATR-normalised gap size as score
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    atr   = (high - low).rolling(14).mean().iloc[-1]
    gap   = abs(curr_diff)
    score = min(1.0, gap / (atr * 0.5 + 1e-9))
    if score < 0.1:
        return None

    if curr_diff > 0:   # bullish
        return "long",  score, {"ema9": e9.iloc[-1], "ema21": e21.iloc[-1], "gap": gap}
    else:               # bearish
        return "short", score, {"ema9": e9.iloc[-1], "ema21": e21.iloc[-1], "gap": gap}


# ── volume spike helper (used by SignalEngine, not a standalone detector) ──

def volume_spike(df: pd.DataFrame, threshold: float = 2.0) -> bool:
    """True if the last bar's volume is > threshold × 20-bar average."""
    vol = df["volume"].astype(float)
    if len(vol) < 21:
        return False
    avg = vol.iloc[-21:-1].mean()
    return vol.iloc[-1] > avg * threshold
