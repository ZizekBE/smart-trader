#!/usr/bin/env python3
"""Replay historical candles through SignalEngine and collect labeled training data.

For each 1h bar from warmup onwards:
  1. Run TrendEngine on preceding daily candles (cached per calendar day)
  2. Run SignalEngine on a rolling 500-bar 1h window
  3. For each SignalEvent, record features + metadata
  4. Label with T+label_horizon bars net return (close-to-close minus 2*fee_rate)

Output: Parquet with one row per signal event. Columns include all SignalEvent.features
keys plus symbol, bar_time, signal_type, source, confidence, raw_score, regime,
net_ret, label.

Usage::

    uv run python scripts/collect_signal_labels.py \\
        --symbols ETH/USDT BTC/USDT \\
        --exchange binance \\
        --label-horizon 20 \\
        --output data/signal_labels/labels.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
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

WARMUP_BARS = 250   # ema_200 + ATR warmup
HOURLY_WINDOW = 500  # 1h bars fed to SignalEngine per step
DAILY_CONTEXT = 120  # 1d bars fed to TrendEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect LightGBM signal labels via historical replay")
    p.add_argument("--symbols", nargs="+", default=["ETH/USDT"])
    p.add_argument("--exchange", default="binance")
    p.add_argument(
        "--label-horizon", type=int, default=20,
        help="Forward 1h bars for labeling (default: 20)",
    )
    p.add_argument(
        "--fee-rate", type=float, default=0.0004,
        help="One-way fee rate; round-trip cost = 2*fee_rate (default: 0.0004)",
    )
    p.add_argument("--min-confidence", type=float, default=0.40)
    p.add_argument(
        "--output", default=None,
        help="Output .parquet path (auto-generated under data/signal_labels/ if omitted)",
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
        else:
            log.warning("no data for %s %s %s — will synthesize if possible", exchange, symbol, tf)

    # Synthesize daily candles from 1h if 1d not available in DB
    if "1d" not in result and "1h" in result:
        df_1d = (
            result["1h"]
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        result["1d"] = df_1d
        log.info("synthesized 1d from 1h for %s %s: %d daily bars", exchange, symbol, len(df_1d))

    # Synthesize 4h candles from 1h if 4h not available
    if "4h" not in result and "1h" in result:
        df_4h = (
            result["1h"]
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        result["4h"] = df_4h
        log.info("synthesized 4h from 1h for %s %s: %d 4h bars", exchange, symbol, len(df_4h))

    return result


def replay_signals(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_4h: pd.DataFrame,
    symbol: str,
    label_horizon: int,
    fee_rate: float,
    min_confidence: float,
) -> pd.DataFrame:
    from smart_trader.data.features.engine import compute_features
    from smart_trader.strategy.signals.engine import SignalEngine
    from smart_trader.strategy.trend.engine import TrendEngine

    trend_engine = TrendEngine()
    signal_engine = SignalEngine()

    # Pre-compute normalized feature matrices (same DatetimeIndex as input)
    log.info("pre-computing feature matrices for %s...", symbol)
    feat_1h = compute_features(df_1h)
    feat_4h = compute_features(df_4h, prefix="4h_")

    records: list[dict] = []
    n = len(df_1h)
    round_trip_cost = 2.0 * fee_rate

    cached_trend_date: object = None
    cached_trend_state = None

    for i in range(WARMUP_BARS, n - label_horizon):
        bar_time = df_1h.index[i]
        bar_date = bar_time.date()

        # Recompute TrendEngine only when calendar day changes
        if bar_date != cached_trend_date:
            df_daily_ctx = df_1d[df_1d.index < bar_time].tail(DAILY_CONTEXT).reset_index()
            if len(df_daily_ctx) >= 60:
                try:
                    cached_trend_state = trend_engine.analyse(df_daily_ctx, symbol=symbol, timeframe="1d")
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

        close_entry = float(df_1h.iloc[i]["close"])
        close_exit = float(df_1h.iloc[i + label_horizon]["close"])
        gross_ret = (close_exit - close_entry) / close_entry

        # Normalized 1h context features at signal bar
        ctx_features: dict = {}
        if bar_time in feat_1h.index:
            ctx_row = feat_1h.loc[bar_time]
            ctx_features = {f"ctx_{k}": float(v) for k, v in ctx_row.items()
                            if not pd.isna(v)}

        # Normalized 4h context features (asof: latest 4h bar before signal)
        feat_4h_idx = feat_4h.index.searchsorted(bar_time, side="right") - 1
        if feat_4h_idx >= 0:
            f4h_row = feat_4h.iloc[feat_4h_idx]
            ctx_features.update({k: float(v) for k, v in f4h_row.items() if not pd.isna(v)})

        # Time features
        time_features = {
            "hour_of_day": bar_time.hour,
            "day_of_week": bar_time.dayofweek,
        }

        for ev in events:
            direction = 1.0 if ev.signal_type == "long" else -1.0
            net_ret = direction * gross_ret - round_trip_cost

            row: dict = {
                "symbol": symbol,
                "bar_time": bar_time,
                "signal_type": ev.signal_type,
                "source": ev.source,
                "confidence": ev.confidence,
                "raw_score": ev.raw_score,
                "regime": ev.regime.name if hasattr(ev.regime, "name") else str(ev.regime),
                "net_ret": net_ret,
                "label": int(net_ret > 0),
            }
            # Add detector features (prefixed to avoid collisions)
            row.update({f"det_{k}": v for k, v in ev.features.items()})
            row.update(ctx_features)
            row.update(time_features)
            records.append(row)

        if (i - WARMUP_BARS) % 500 == 0:
            log.info("%s  bar %d/%d  signals_so_far=%d", symbol, i, n, len(records))

    df_out = pd.DataFrame(records)
    log.info("collected %d signal records for %s", len(df_out), symbol)
    return df_out


async def main() -> None:
    args = parse_args()

    all_frames: list[pd.DataFrame] = []
    for symbol in args.symbols:
        data = await load_candles(symbol, args.exchange)
        if "1h" not in data or "1d" not in data:
            log.warning("missing 1h or 1d candles for %s, skipping", symbol)
            continue
        df = replay_signals(
            df_1h=data["1h"],
            df_1d=data["1d"],
            df_4h=data.get("4h", data["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()),
            symbol=symbol,
            label_horizon=args.label_horizon,
            fee_rate=args.fee_rate,
            min_confidence=args.min_confidence,
        )
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        log.error("no signals collected — check DB connectivity and candle coverage")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    if args.output is None:
        out_dir = ROOT / "data" / "signal_labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        syms = "_".join(s.replace("/", "") for s in args.symbols)
        args.output = str(out_dir / f"labels_{syms}_h{args.label_horizon}_{ts}.parquet")

    combined.to_parquet(args.output, index=False)
    log.info("saved %d rows → %s", len(combined), args.output)

    if "label" in combined.columns and len(combined) > 0:
        rate = combined["label"].mean()
        log.info(
            "label distribution: positive=%.1f%%  negative=%.1f%%  total=%d",
            rate * 100, (1 - rate) * 100, len(combined),
        )
        if "signal_type" in combined.columns:
            print("\n--- by signal_type ---")
            print(combined.groupby("signal_type")["label"].agg(count="count", win_rate="mean").to_string())
        if "source" in combined.columns:
            print("\n--- by source ---")
            print(combined.groupby("source")["label"].agg(count="count", win_rate="mean").to_string())


if __name__ == "__main__":
    asyncio.run(main())
