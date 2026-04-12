"""Shadow mode — validate trained RL agent on recent data without executing.

Replays the latest N days of DB candles through the model, simulating
portfolio state and logging every decision.  Produces a summary with
return, Sharpe, max drawdown, and a per-bar decision log.

Usage::

    cd python
    uv run python scripts/run_shadow.py --symbol BTC/USDT --days 7
    uv run python scripts/run_shadow.py --symbol ETH/USDT --days 14 --log-file shadow_eth.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL Shadow Mode Validation")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--model", default=None,
                   help="Path to .pt checkpoint (default: auto-detect v7_long best_agent)")
    p.add_argument("--days", type=int, default=7,
                   help="Number of recent days to replay")
    p.add_argument("--exchange", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--log-file", default=None,
                   help="JSONL file for per-bar decision log")
    p.add_argument("--max-leverage", type=float, default=3.0)
    return p.parse_args()


def auto_detect_model(symbol: str) -> str:
    safe = symbol.replace("/", "_")
    candidates = [
        ROOT / f"checkpoints/v7_long/{safe}/best_agent.pt",
        ROOT / f"checkpoints/v7/{safe}/best_agent.pt",
        ROOT / f"checkpoints/v6/{safe}/best_agent.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No checkpoint found for {symbol}. Tried: {[str(c) for c in candidates]}"
    )


async def load_recent_data(
    symbol: str, days: int, exchange: str | None = None,
) -> dict[str, pd.DataFrame]:
    from smart_trader.data.storage.database import get_engine
    from sqlalchemy import text

    engine = get_engine()
    since = datetime.now(timezone.utc) - timedelta(days=days + 1)
    data: dict[str, pd.DataFrame] = {}

    for tf in ("1m", "1h", "4h"):
        where = "symbol = :sym AND timeframe = :tf AND time >= :since"
        params: dict = {"sym": symbol, "tf": tf, "since": since}
        if exchange:
            where += " AND exchange = :ex"
            params["ex"] = exchange

        sql = text(f"""
            SELECT time, open::double precision, high::double precision,
                   low::double precision, close::double precision,
                   volume::double precision
            FROM candles WHERE {where} ORDER BY time ASC
        """)

        async with engine.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.fetchall()

        if not rows:
            log.warning("no_data", symbol=symbol, timeframe=tf)
            continue

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df = df.set_index("time")
        data[tf] = df
        log.info("loaded", symbol=symbol, tf=tf, rows=len(df),
                 start=str(df.index[0]), end=str(df.index[-1]))

    return data


def run_shadow(args: argparse.Namespace, data: dict[str, pd.DataFrame]) -> None:
    import torch
    from smart_trader.agent.meta_controller import MetaController
    from smart_trader.agent.observation import ObservationBuilder, PortfolioSnapshot
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.spaces import SpaceConfig

    model_path = args.model or auto_detect_model(args.symbol)
    log.info("loading_model", path=model_path)

    ckpt = torch.load(model_path, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    obs_dim = cfg["obs_dim"]
    d_model = cfg["d_model"]
    n_layers = cfg.get("n_layers", 2)
    n_heads = cfg.get("n_heads", 4)
    lookback = cfg.get("lookback", 1)
    context_dim = cfg.get("context_dim", 0)

    agent = MetaController(
        obs_dim=obs_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, device=args.device,
        lookback=lookback, context_dim=context_dim,
    )
    agent.load(model_path)
    agent.set_deterministic(True)

    feat_cfg = FeatureConfig()
    tf_features: dict[str, pd.DataFrame] = {}
    for tf, df in data.items():
        tf_features[tf] = compute_features(df, feat_cfg, prefix=f"{tf}_")

    sample_cols = len(list(tf_features.values())[0].columns)
    n_tf = len(tf_features)
    space_cfg = SpaceConfig(n_timeframes=n_tf, features_per_tf=sample_cols,
                            lookback=lookback)
    obs_builder = ObservationBuilder(space_cfg)
    obs_builder.reset()

    expected_dim = obs_builder.obs_dim
    if expected_dim != obs_dim:
        log.warning("obs_dim_mismatch", expected=expected_dim, model=obs_dim,
                    hint="Padding will handle the difference")

    primary_tf = "1m"
    primary_df = data[primary_tf]
    primary_feat = tf_features[primary_tf]

    cash = args.initial_cash
    position_qty = 0.0
    entry_price = 0.0
    peak_value = cash
    total_trades = 0
    wins = 0
    pnl_history: list[float] = []
    records: list[dict] = []

    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w")

    print(f"\n{'═' * 60}")
    print(f"  Shadow Mode — {args.symbol}")
    print(f"{'═' * 60}")
    print(f"  Model:      {Path(model_path).name}")
    print(f"  Lookback:   {lookback}")
    print(f"  Period:     {primary_df.index[0]} → {primary_df.index[-1]}")
    print(f"  Bars:       {len(primary_df):,}")
    print(f"  Initial:    ${cash:,.2f}")
    print(f"{'═' * 60}\n")

    t0 = time.monotonic()
    prev_position_target = 0.0

    for i in range(len(primary_feat)):
        row_tf: dict[str, np.ndarray] = {}
        ts = primary_feat.index[i]

        for tf, feat_df in tf_features.items():
            mask = feat_df.index <= ts
            if mask.any():
                row_tf[tf] = feat_df.loc[mask].iloc[-1].to_numpy(dtype=np.float32, na_value=0.0)
            else:
                row_tf[tf] = np.zeros(len(feat_df.columns), dtype=np.float32)

        price = float(primary_df.iloc[i]["close"])
        notional = position_qty * price
        total_value = cash + notional
        unrealized_pnl = position_qty * (price - entry_price) if position_qty != 0 else 0.0

        portfolio = PortfolioSnapshot(
            position_fraction=notional / (total_value + 1e-9),
            unrealized_pnl_frac=unrealized_pnl / (total_value + 1e-9),
            margin_frac=abs(notional) / (total_value + 1e-9) / args.max_leverage,
            cash_frac=cash / (total_value + 1e-9),
            leverage_frac=abs(notional) / (total_value + 1e-9),
            drawdown=(peak_value - total_value) / (peak_value + 1e-9),
        )

        obs = obs_builder.build(
            tf_features=row_tf,
            portfolio=portfolio,
            timestamp=pd.Timestamp(ts),
        )

        if len(obs) < obs_dim:
            obs = np.pad(obs, (0, obs_dim - len(obs)))
        elif len(obs) > obs_dim:
            obs = obs[:obs_dim]

        action, _, value = agent.act(obs, deterministic=True)
        target_pos = float(np.atleast_1d(action.get("position", 0))[0])
        risk_budget = float(np.atleast_1d(action.get("risk_budget", 0.02))[0])
        hold_bars = int(action.get("hold_bars", 0))

        target_notional = target_pos * total_value
        delta_notional = target_notional - notional
        scale = max(abs(notional), abs(target_notional), total_value)
        dead_zone = scale * 0.10

        trade_executed = False
        if abs(delta_notional) > dead_zone:
            delta_qty = delta_notional / price
            if position_qty != 0 and np.sign(position_qty + delta_qty) != np.sign(position_qty):
                realized = position_qty * (price - entry_price)
                pnl_history.append(realized)
                if realized > 0:
                    wins += 1
                total_trades += 1
                cash += notional + realized
                position_qty = 0.0
                entry_price = 0.0

                remaining_notional = target_notional
                if abs(remaining_notional) > dead_zone:
                    position_qty = remaining_notional / price
                    entry_price = price
                    cash -= remaining_notional
            else:
                position_qty += delta_qty
                if abs(position_qty) < 1e-12:
                    position_qty = 0.0
                    entry_price = 0.0
                else:
                    entry_price = (entry_price * (position_qty - delta_qty) + price * delta_qty) / position_qty if position_qty != 0 else price
                cash -= delta_notional
            trade_executed = True

        total_value = cash + position_qty * price
        peak_value = max(peak_value, total_value)

        record = {
            "time": str(ts),
            "price": round(price, 2),
            "target_pos": round(target_pos, 4),
            "risk_budget": round(risk_budget, 4),
            "hold_bars": hold_bars,
            "value_est": round(value, 4),
            "portfolio_value": round(total_value, 2),
            "position_frac": round(notional / (total_value + 1e-9), 4),
            "traded": trade_executed,
        }
        records.append(record)

        if log_file:
            log_file.write(json.dumps(record) + "\n")

        if i > 0 and i % 10000 == 0:
            ret = (total_value - args.initial_cash) / args.initial_cash * 100
            log.info("progress", bar=f"{i}/{len(primary_feat)}",
                     value=f"${total_value:,.2f}", ret=f"{ret:+.2f}%",
                     trades=total_trades)

    if log_file:
        log_file.close()

    elapsed = time.monotonic() - t0

    if position_qty != 0:
        final_price = float(primary_df.iloc[-1]["close"])
        realized = position_qty * (final_price - entry_price)
        pnl_history.append(realized)
        if realized > 0:
            wins += 1
        total_trades += 1

    final_value = cash + position_qty * float(primary_df.iloc[-1]["close"])
    total_return = (final_value - args.initial_cash) / args.initial_cash * 100
    max_dd = max(
        (r["portfolio_value"] for r in records),
        default=args.initial_cash,
    )
    dd_series = []
    running_peak = args.initial_cash
    for r in records:
        running_peak = max(running_peak, r["portfolio_value"])
        dd_series.append((running_peak - r["portfolio_value"]) / running_peak)
    max_dd_pct = max(dd_series) * 100 if dd_series else 0.0

    values = [r["portfolio_value"] for r in records]
    if len(values) > 1:
        returns = np.diff(values) / np.array(values[:-1])
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 60))
    else:
        sharpe = 0.0

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0

    trade_count = sum(1 for r in records if r["traded"])
    position_bars = sum(1 for r in records if abs(r["position_frac"]) > 0.01)
    position_pct = position_bars / len(records) * 100 if records else 0.0

    print(f"\n{'═' * 60}")
    print(f"  Shadow Mode Results — {args.symbol}")
    print(f"{'═' * 60}")
    print(f"  Period:        {records[0]['time']} → {records[-1]['time']}")
    print(f"  Bars:          {len(records):,}")
    print(f"  Elapsed:       {elapsed:.1f}s")
    print(f"{'─' * 60}")
    print(f"  Total return:  {total_return:+.2f}%")
    print(f"  Final value:   ${final_value:,.2f}")
    print(f"  Sharpe ratio:  {sharpe:.2f}")
    print(f"  Max drawdown:  {max_dd_pct:.2f}%")
    print(f"{'─' * 60}")
    print(f"  Total trades:  {total_trades}")
    print(f"  Trade signals: {trade_count}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Position time: {position_pct:.1f}%")
    if pnl_history:
        print(f"  Avg PnL/trade: ${np.mean(pnl_history):,.2f}")
    print(f"{'═' * 60}\n")

    if args.log_file:
        print(f"  Decision log saved to: {args.log_file}")


def main() -> None:
    from smart_trader.core.env_files import load_repo_env_files
    load_repo_env_files()

    args = parse_args()
    data = asyncio.run(load_recent_data(args.symbol, args.days, args.exchange))

    if "1m" not in data:
        log.error("no_1m_data", hint="Need 1m candles for shadow replay")
        return

    run_shadow(args, data)


if __name__ == "__main__":
    main()
