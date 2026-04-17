#!/usr/bin/env python3
"""ML signal-filter shadow validation.

Replays the latest N days of DB candles through both the unfiltered rule
engine and the LightGBM-filtered engine side-by-side.  No orders are placed.
Produces a per-signal JSONL log and a printed comparison summary.

Methodology
-----------
For each 1h bar (from ``--days`` ago up to ``horizon`` bars before now):
  1. Run TrendEngine on the daily context window (cached per calendar day).
  2. Run SignalEngine on a 500-bar rolling 1h window.
  3. For each candidate signal, ask the LightGBM filter whether it would pass.
  4. Record: time, symbol, signal metadata, filter decision, proba, direction.
  5. Compute T+``--horizon`` forward return from the 1h close series.

Signals fired in the last ``horizon`` bars cannot be labelled yet — they are
logged with ``forward_ret=null`` and ``label=null``.

Usage::

    uv run python scripts/run_ml_filter_shadow.py \\
        --symbols ETH/USDT BTC/USDT \\
        --exchange binance \\
        --days 30 \\
        --model models/signal_filter \\
        --threshold 0.55 \\
        --horizon 8 \\
        --log-file shadow_ml_filter.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
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

WARMUP_BARS    = 250
HOURLY_WINDOW  = 500
DAILY_CONTEXT  = 120


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML signal-filter shadow validation")
    p.add_argument("--symbols",   nargs="+", default=["ETH/USDT"])
    p.add_argument("--exchange",  default="binance")
    p.add_argument("--days",      type=int, default=30, help="Recent days to replay")
    p.add_argument("--model",     default="models/signal_filter")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--min-votes", type=int,   default=1,
                   help="Phase-1.2 vote gate applied to BOTH pipelines")
    p.add_argument("--horizon",   type=int,   default=8,
                   help="Forward bars for outcome labelling (must match training)")
    p.add_argument("--fee-rate",  type=float, default=0.0004)
    p.add_argument("--log-file",  default=None,
                   help="JSONL output (auto-named under data/shadow/ if omitted)")
    return p.parse_args()


async def load_candles(symbol: str, exchange: str, days: int) -> dict[str, pd.DataFrame]:
    from sqlalchemy import text
    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    # TrendEngine needs ≥60 daily bars; always load 130+ days of 1h history
    history_days = max(days + WARMUP_BARS // 24 + 5, days + 130)
    since_1h = datetime.now(timezone.utc) - timedelta(days=history_days)
    since_4h = datetime.now(timezone.utc) - timedelta(days=days + 60)

    result: dict[str, pd.DataFrame] = {}
    queries = {
        "1h": (since_1h, "1h"),
        "4h": (since_4h, "4h"),
    }
    for key, (since, tf) in queries.items():
        sql = text("""
            SELECT time, open::double precision, high::double precision,
                   low::double precision, close::double precision,
                   volume::double precision
            FROM candles
            WHERE exchange = :ex AND symbol = :sym AND timeframe = :tf
              AND time >= :since
            ORDER BY time ASC
        """)
        async with engine.connect() as conn:
            rows = (await conn.execute(sql, {"ex": exchange, "sym": symbol,
                                            "tf": tf, "since": since})).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
            result[key] = df
            log.info("loaded %s %s %s: %d bars", exchange, symbol, tf, len(df))
        else:
            log.warning("no %s data for %s %s", tf, exchange, symbol)

    # Synthesise 4h from 1h if missing
    if "4h" not in result and "1h" in result:
        result["4h"] = (
            result["1h"]
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
        )
        log.info("synthesised 4h from 1h for %s", symbol)

    # Synthesise 1d from 1h (for TrendEngine)
    if "1h" in result:
        result["1d"] = (
            result["1h"]
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
        )

    return result


def load_filter(model_dir: Path, threshold: float):
    from smart_trader.strategy.signal_filter import SignalFilter
    return SignalFilter.from_model_dir(model_dir, threshold=threshold)


def run_shadow(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    symbol: str,
    signal_filter,
    threshold: float,
    min_votes: int,
    horizon: int,
    fee_rate: float,
    replay_since: datetime,
) -> list[dict]:
    from smart_trader.data.features.engine import compute_features
    from smart_trader.strategy.signals.engine import SignalEngine
    from smart_trader.strategy.trend.engine import TrendEngine

    trend_engine  = TrendEngine()
    signal_engine = SignalEngine()

    feat_1h = compute_features(df_1h)
    feat_4h = compute_features(df_4h, prefix="4h_")

    n   = len(df_1h)
    records: list[dict] = []
    cached_trend_date = None
    cached_trend_state = None

    # Find first index at or after replay_since
    replay_idx = int(df_1h.index.searchsorted(replay_since, side="left"))
    start_idx  = max(WARMUP_BARS, replay_idx)

    log.info("%s: replaying bars %d – %d (of %d total)", symbol, start_idx, n - horizon - 1, n)

    for i in range(start_idx, n - horizon):
        bar_time = df_1h.index[i]
        bar_date = bar_time.date()

        # Daily TrendEngine (cached)
        if bar_date != cached_trend_date:
            df_daily_ctx = df_1d[df_1d.index < bar_time].tail(DAILY_CONTEXT).reset_index()
            if len(df_daily_ctx) >= 60:
                try:
                    cached_trend_state = trend_engine.analyse(df_daily_ctx, symbol=symbol, timeframe="1d")
                    cached_trend_date  = bar_date
                except Exception as exc:
                    log.debug("TrendEngine error at %s: %s", bar_time, exc)
                    cached_trend_state = None

        if cached_trend_state is None:
            continue

        df_ctx = df_1h.iloc[max(0, i - HOURLY_WINDOW + 1):i + 1].reset_index()
        try:
            events = signal_engine.analyse(
                df=df_ctx,
                symbol=symbol,
                timeframe="1h",
                trend_state=cached_trend_state,
                min_votes=min_votes,
            )
        except Exception as exc:
            log.debug("SignalEngine error at %s: %s", bar_time, exc)
            continue

        if not events:
            continue

        # Apply filter to get per-event proba decisions
        df_1h_window = df_1h.iloc[max(0, i - HOURLY_WINDOW + 1):i + 1]
        df_4h_window = df_4h[df_4h.index <= bar_time]

        # Forward return
        close_entry = float(df_1h.iloc[i]["close"])
        close_exit  = float(df_1h.iloc[i + horizon]["close"])
        gross_ret   = (close_exit - close_entry) / close_entry

        # Compute probabilities for all events at once
        events_proba: list[float] = []
        if signal_filter is not None and events:
            # Temporarily get proba by running filter internals
            # We replicate the proba computation here to get per-event scores
            proba_list = _get_probas(signal_filter, events, df_1h_window, df_4h_window, symbol)
            events_proba = proba_list
        else:
            events_proba = [1.0] * len(events)  # no filter → always pass

        for ev, proba in zip(events, events_proba):
            direction    = 1.0 if ev.signal_type == "long" else -1.0
            net_ret      = direction * gross_ret - 2.0 * fee_rate
            filter_pass  = proba >= threshold

            records.append({
                "time":        bar_time.isoformat(),
                "symbol":      symbol,
                "signal_type": ev.signal_type,
                "source":      ev.source,
                "regime":      ev.regime.name if hasattr(ev.regime, "name") else str(ev.regime),
                "confidence":  round(float(ev.confidence), 4),
                "vote_count":  int(ev.features.get("vote_count", 1)),
                "proba":       round(float(proba), 4),
                "filter_pass": filter_pass,
                "forward_ret": round(float(net_ret), 6),
                "label":       int(net_ret > 0),
            })

    return records


def _get_probas(
    signal_filter,
    events: list,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    symbol: str,
) -> list[float]:
    """Extract per-event probabilities from the filter without dropping events."""
    import pickle, json, numpy as np
    from smart_trader.data.features.engine import compute_features

    feat_1h = compute_features(df_1h) if not df_1h.empty else pd.DataFrame()
    feat_4h = compute_features(df_4h, prefix="4h_") if not df_4h.empty else pd.DataFrame()

    _CAT_COLS  = ["signal_type", "source", "regime", "symbol"]
    _RAW_PRICE = {"close", "upper", "mid", "ema20", "ema50", "low", "high", "lower", "vwap"}

    rows: list[dict] = []
    for ev in events:
        bar_time = ev.created_at
        ctx: dict = {}

        if not feat_1h.empty:
            idx = feat_1h.index.searchsorted(bar_time, side="right") - 1
            if idx >= 0:
                ctx.update({f"ctx_{k}": float(v) for k, v in feat_1h.iloc[idx].items()
                            if not pd.isna(v)})

        if not feat_4h.empty:
            idx = feat_4h.index.searchsorted(bar_time, side="right") - 1
            if idx >= 0:
                ctx.update({k: float(v) for k, v in feat_4h.iloc[idx].items()
                            if not pd.isna(v)})

        det = {f"det_{k}": v for k, v in ev.features.items()
               if f"det_{k[4:]}" not in _RAW_PRICE}

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
    encoders     = signal_filter._encoders
    feature_cols = signal_filter._feature_cols

    for col in _CAT_COLS:
        if col not in df.columns:
            df[col] = ""
        le = encoders.get(col)
        if le is not None:
            mapping = {v: i for i, v in enumerate(le.classes_)}
            df[f"{col}_enc"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_enc"] = -1

    X = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    col_index = {c: i for i, c in enumerate(feature_cols)}
    for col in df.columns:
        if col in col_index:
            X[:, col_index[col]] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

    return list(signal_filter._booster.predict(X).tolist())


def summarise(records: list[dict]) -> None:
    if not records:
        log.warning("no signals recorded")
        return

    df = pd.DataFrame(records)
    total = len(df)

    base  = df                        # all detected signals
    filt  = df[df["filter_pass"]]     # filter-accepted signals

    def metrics(d: pd.DataFrame, name: str) -> None:
        if d.empty:
            log.info("%s: 0 signals", name)
            return
        wr  = (d["label"] == 1).mean()
        ret = d["forward_ret"].mean()
        log.info(
            "%s: n=%d  WR=%.1f%%  MeanRet=%.5f",
            name, len(d), wr * 100, ret,
        )

    print("\n" + "=" * 60)
    print(f"  ML Filter Shadow — {total} candidate signals")
    print("=" * 60)
    metrics(base,  "Baseline (all)  ")
    metrics(filt,  "Filtered (pass) ")
    kept_pct = len(filt) / max(total, 1) * 100
    print(f"  Filter kept {len(filt)}/{total} = {kept_pct:.1f}%  threshold={df['proba'].mean():.3f} avg-proba")
    print("=" * 60)

    if "source" in df.columns:
        print("\n--- by source ---")
        print(
            df.groupby("source")[["label", "filter_pass"]]
            .agg(signals=("label", "count"), win_rate=("label", "mean"),
                 filter_rate=("filter_pass", "mean"))
            .round(3)
            .to_string()
        )


async def main() -> None:
    args = parse_args()

    model_dir = ROOT / args.model
    if not (model_dir / "lgb_v1.model").exists():
        log.error("model not found at %s — run train_signal_filter.py first", model_dir)
        sys.exit(1)

    signal_filter = load_filter(model_dir, args.threshold)

    # Auto-name log file
    if args.log_file is None:
        out_dir = ROOT / "data" / "shadow"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        syms = "_".join(s.replace("/", "") for s in args.symbols)
        log_path = out_dir / f"shadow_ml_{syms}_{ts}.jsonl"
    else:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    replay_since = datetime.now(timezone.utc) - timedelta(days=args.days)

    all_records: list[dict] = []
    for symbol in args.symbols:
        data = await load_candles(symbol, args.exchange, args.days)
        if "1h" not in data or "1d" not in data:
            log.warning("missing candles for %s, skipping", symbol)
            continue

        records = run_shadow(
            df_1h         = data["1h"],
            df_4h         = data.get("4h", pd.DataFrame()),
            df_1d         = data["1d"],
            symbol        = symbol,
            signal_filter = signal_filter,
            threshold     = args.threshold,
            min_votes     = args.min_votes,
            horizon       = args.horizon,
            fee_rate      = args.fee_rate,
            replay_since  = replay_since,
        )
        log.info("%s: %d signal records", symbol, len(records))
        all_records.extend(records)

    if not all_records:
        log.error("no signal records produced")
        sys.exit(1)

    with open(log_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    log.info("log saved → %s  (%d records)", log_path, len(all_records))

    summarise(all_records)


if __name__ == "__main__":
    asyncio.run(main())
