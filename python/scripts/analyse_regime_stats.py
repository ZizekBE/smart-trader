"""T-L1-01-2: Per-regime backtest statistics.

Loads historical OHLCV data, labels each primary-timeframe bar with a
market regime via TrendEngine (same logic as regime_features.py), then
computes forward-return statistics per state to verify that states carry
distinct business meaning.

Usage::
    cd python
    uv run python scripts/analyse_regime_stats.py \
        --symbols ETH/USDT --exchange binance \
        --primary-tf 1h --forward-bars 1 4 24
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.WARNING)

_REGIME_ORDER = [
    "bear_trending", "bear_ranging", "distribution",
    "accumulation", "bull_ranging", "bull_trending",
]


async def _load(symbols: list[str], exchange: str) -> dict[str, dict[str, pd.DataFrame]]:
    from smart_trader.core.env_files import load_repo_env_files
    load_repo_env_files()

    from sqlalchemy import text
    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for sym in symbols:
        frames: dict[str, pd.DataFrame] = {}
        for tf in ("1h", "4h", "1d"):
            where = "symbol = :sym AND timeframe = :tf AND exchange = :ex"
            params = {"sym": sym, "tf": tf, "ex": exchange}
            sql = text(f"""
                SELECT time, open::double precision, high::double precision,
                       low::double precision, close::double precision,
                       volume::double precision
                FROM candles WHERE {where} ORDER BY time ASC
            """)
            async with engine.connect() as conn:
                rows = (await conn.execute(sql, params)).fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                df = df.set_index("time")
                frames[tf] = df
        result[sym] = frames

    return result


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


def _compute_labels(
    daily_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    symbol: str,
    stride: int = 6,
) -> pd.Series:
    from smart_trader.strategy.trend.engine import TrendEngine
    engine = TrendEngine()

    MIN_BARS = 60
    n = len(daily_df)
    labels: dict[pd.Timestamp, str] = {}

    for i in range(MIN_BARS, n):
        if (i - MIN_BARS) % stride != 0 and i != n - 1:
            continue
        ts = daily_df.index[i]
        try:
            state = engine.analyse(daily_df.iloc[: i + 1], symbol, "1d")
            labels[ts] = str(state.regime).lower()
        except Exception:
            labels[ts] = "unknown"

    if not labels:
        return pd.Series("unknown", index=primary_df.index, name="regime")

    s = pd.Series(labels, name="regime")
    s.index = pd.to_datetime(s.index, utc=True)

    combined = primary_df.index.union(s.index)
    s = s.reindex(combined).ffill().bfill()
    return s.reindex(primary_df.index).fillna("unknown")


def _stats_for_group(returns: pd.Series) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "mean_ret": np.nan, "std_ret": np.nan,
                "sharpe": np.nan, "win_rate": np.nan, "median_ret": np.nan}
    mean = returns.mean()
    std = returns.std()
    sharpe = (mean / std * np.sqrt(252)) if std > 1e-9 else np.nan
    return {
        "n": n,
        "mean_ret_%": round(mean * 100, 4),
        "std_ret_%": round(std * 100, 4),
        "sharpe_ann": round(sharpe, 3),
        "win_rate_%": round((returns > 0).mean() * 100, 1),
        "median_ret_%": round(returns.median() * 100, 4),
    }


def analyse(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    forward_bars: list[int],
    primary_tf: str,
) -> None:
    primary_df = frames.get(primary_tf)
    hourly_df = frames.get("1h")
    daily_df = frames.get("1d")

    if primary_df is None or primary_df.empty:
        print(f"[{symbol}] no primary data for {primary_tf}")
        return

    primary_df = _ensure_utc(primary_df)

    if daily_df is None or len(daily_df) < 60:
        if hourly_df is not None and not hourly_df.empty:
            hourly_df = _ensure_utc(hourly_df)
            daily_df = _resample_to_daily(hourly_df)
        else:
            print(f"[{symbol}] insufficient daily data")
            return
    else:
        daily_df = _ensure_utc(daily_df)

    print(f"\nComputing regime labels for {symbol} ({len(daily_df)} daily bars)…")
    labels = _compute_labels(daily_df, primary_df, symbol)

    closes = primary_df["close"]
    print(f"\n{'═'*68}")
    print(f"  Regime Statistics — {symbol} / {primary_tf}  "
          f"({len(primary_df):,} bars, {labels.value_counts().to_dict()})")
    print(f"{'═'*68}")

    for fwd in forward_bars:
        fwd_ret = closes.pct_change(fwd).shift(-fwd)
        print(f"\n  Forward {fwd}-bar return:")
        print(f"  {'Regime':<18} {'n':>6} {'mean%':>8} {'std%':>8} "
              f"{'sharpe':>8} {'win%':>7} {'median%':>9}")
        print(f"  {'─'*66}")

        rows = []
        for regime in _REGIME_ORDER + ["unknown"]:
            mask = labels == regime
            if mask.sum() == 0:
                continue
            st = _stats_for_group(fwd_ret[mask].dropna())
            rows.append((regime, st))

        # sort by mean_ret
        rows.sort(key=lambda x: x[1].get("mean_ret_%", 0) or 0, reverse=True)

        for regime, st in rows:
            if st["n"] == 0:
                continue
            print(
                f"  {regime:<18} {st['n']:>6,} "
                f"{st['mean_ret_%']:>8.3f} {st['std_ret_%']:>8.3f} "
                f"{st['sharpe_ann']:>8.3f} {st['win_rate_%']:>7.1f} "
                f"{st['median_ret_%']:>9.4f}"
            )

        # overall
        overall = _stats_for_group(fwd_ret.dropna())
        print(f"  {'─'*66}")
        print(
            f"  {'[overall]':<18} {overall['n']:>6,} "
            f"{overall['mean_ret_%']:>8.3f} {overall['std_ret_%']:>8.3f} "
            f"{overall['sharpe_ann']:>8.3f} {overall['win_rate_%']:>7.1f} "
            f"{overall['median_ret_%']:>9.4f}"
        )

    print(f"\n{'═'*68}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ETH/USDT"])
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--primary-tf", default="1h")
    ap.add_argument("--forward-bars", nargs="+", type=int, default=[1, 4, 24])
    args = ap.parse_args()

    all_frames = asyncio.run(_load(args.symbols, args.exchange))
    for sym in args.symbols:
        analyse(sym, all_frames[sym], args.forward_bars, args.primary_tf)


if __name__ == "__main__":
    main()
