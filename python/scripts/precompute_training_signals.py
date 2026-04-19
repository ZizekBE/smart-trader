#!/usr/bin/env python3
"""Precompute rule-engine signal series for hybrid RL training.

Replays the rule engine over historical candles and outputs a compact Parquet
with one row per primary-TF bar containing the signal at that bar:
  {ts (index), direction (-1/0/1), confidence (0-1)}

Bars with no active signal get direction=0, confidence=0.0.

This Parquet is then passed to train_rl_agent.py --hybrid-mode --signal-parquet
to construct the signal_data dict for MarketEnv.

Usage::

    cd python && uv run python scripts/precompute_training_signals.py \\
        --symbols ETH/USDT BTC/USDT --exchange binance \\
        --output data/hybrid_signals/signals.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WARMUP_BARS = 250
HOURLY_WINDOW = 500
DAILY_CONTEXT = 120
PRIMARY_TF = "1h"  # env primary timeframe for hybrid training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute hybrid RL training signals")
    p.add_argument("--symbols", nargs="+", default=["ETH/USDT"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--min-confidence", type=float, default=0.40)
    p.add_argument(
        "--output", default=None,
        help="Output .parquet path (auto-generated under data/hybrid_signals/ if omitted)",
    )
    return p.parse_args()


async def load_candles(symbol: str, exchange: str) -> dict[str, pd.DataFrame]:
    from sqlalchemy import text
    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    result: dict[str, pd.DataFrame] = {}
    for tf in ("1h", "4h", "1d"):
        sql = text("""
            SELECT time, open::double precision, high::double precision,
                   low::double precision, close::double precision,
                   volume::double precision
            FROM candles
            WHERE exchange = :ex AND symbol = :sym AND timeframe = :tf
            ORDER BY time ASC
        """)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, {"ex": exchange, "sym": symbol, "tf": tf})).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            result[tf] = df
            log.info("loaded %s %s %s: %d rows", exchange, symbol, tf, len(df))

    if "1d" not in result and "1h" in result:
        result["1d"] = (
            result["1h"]
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        log.info("synthesized 1d from 1h for %s: %d bars", symbol, len(result["1d"]))

    return result


def build_signal_series(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    symbol: str,
    min_confidence: float,
) -> pd.DataFrame:
    """Replay signal engine over df_1h, return DataFrame(direction, confidence) at 1h timestamps."""
    from smart_trader.strategy.signals.engine import SignalEngine
    from smart_trader.strategy.trend.engine import TrendEngine

    trend_engine = TrendEngine()
    signal_engine = SignalEngine()

    n = len(df_1h)
    timestamps = df_1h.index.tolist()
    directions = [0.0] * n
    confidences = [0.0] * n

    cached_trend_date: object = None
    cached_trend_state = None

    for i in range(WARMUP_BARS, n):
        bar_time = df_1h.index[i]
        bar_date = bar_time.date()

        if bar_date != cached_trend_date:
            df_daily_ctx = df_1d[df_1d.index < bar_time].tail(DAILY_CONTEXT).reset_index()
            if len(df_daily_ctx) >= 60:
                try:
                    cached_trend_state = trend_engine.analyse(
                        df_daily_ctx, symbol=symbol, timeframe="1d",
                    )
                    cached_trend_date = bar_date
                except Exception as exc:
                    log.debug("TrendEngine error at %s: %s", bar_time, exc)
                    cached_trend_state = None

        if cached_trend_state is None:
            continue

        df_1h_ctx = df_1h.iloc[max(0, i - HOURLY_WINDOW + 1):i + 1].reset_index()
        try:
            events = signal_engine.analyse(
                df=df_1h_ctx,
                symbol=symbol,
                timeframe="1h",
                trend_state=cached_trend_state,
                vol_state=None,
                min_confidence=min_confidence,
            )
        except Exception as exc:
            log.debug("SignalEngine error at %s: %s", bar_time, exc)
            continue

        if not events:
            continue

        # Take the strongest event if multiple fire on the same bar
        best = max(events, key=lambda e: e.confidence)
        direction_val = 1.0 if best.signal_type.endswith("long") or "long" in best.signal_type.lower() else -1.0
        directions[i] = direction_val
        confidences[i] = float(best.confidence)

    result = pd.DataFrame(
        {"direction": directions, "confidence": confidences},
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
    )
    signal_bars = (result["direction"] != 0).sum()
    log.info(
        "signal_series done: symbol=%s  total=%d  signal_bars=%d  rate=%.1f%%",
        symbol, n, signal_bars, 100.0 * signal_bars / max(n, 1),
    )
    return result


async def main_async(args: argparse.Namespace) -> None:
    out_path = args.output
    if out_path is None:
        sym_tag = "_".join(s.replace("/", "") for s in args.symbols)
        out_dir = ROOT / "data" / "hybrid_signals"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"signals_{sym_tag}_{args.exchange}.parquet")

    all_frames: list[pd.DataFrame] = []

    for symbol in args.symbols:
        log.info("processing %s...", symbol)
        candles = await load_candles(symbol, args.exchange)
        df_1h = candles.get("1h")
        df_1d = candles.get("1d")
        if df_1h is None or df_1h.empty:
            log.error("no 1h data for %s — skipping", symbol)
            continue
        if df_1d is None or df_1d.empty:
            log.error("no 1d data for %s — skipping", symbol)
            continue

        sig_df = build_signal_series(df_1h, df_1d, symbol, args.min_confidence)
        sig_df["symbol"] = symbol
        all_frames.append(sig_df)

    if not all_frames:
        log.error("no signal data produced — exiting")
        sys.exit(1)

    combined = pd.concat(all_frames)
    combined.to_parquet(out_path)
    total_signals = (combined["direction"] != 0).sum()
    log.info(
        "saved to %s  rows=%d  total_signal_bars=%d",
        out_path, len(combined), total_signals,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
