"""RL agent training script — end-to-end with real DB data.

Loads historical candles from TimescaleDB, builds a Gymnasium MarketEnv,
and trains the hierarchical Meta Controller using PPO.

Usage::

    cd python && uv run python scripts/train_rl_agent.py
    uv run python scripts/train_rl_agent.py --symbol BTC/USDT --timesteps 20000
    uv run python scripts/train_rl_agent.py --walk-forward
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
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
    p = argparse.ArgumentParser(description="Train RL Meta Controller")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                    help="One or more symbols (e.g. BTC/USDT ETH/USDT)")
    p.add_argument("--timesteps", type=int, default=10_000)
    p.add_argument("--d-model", type=int, default=64, help="Transformer hidden dim (smaller = faster)")
    p.add_argument("--n-layers", type=int, default=1, help="Transformer layers")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512, help="PPO rollout length")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-episode", type=int, default=2000, help="Max episode bars")
    p.add_argument("--lookback", type=int, default=30,
                    help="Bars of history in the observation sequence (1 = legacy flat)")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--exchange", default=None, help="Filter by exchange (binance, gateio)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--per-symbol", action="store_true",
                    help="Train a separate model for each symbol")
    return p.parse_args()


async def load_data(
    symbols: list[str], exchange: str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load candle data for one or more symbols.

    Returns ``{symbol: {timeframe: DataFrame}}``.
    """
    from smart_trader.data.storage.database import get_engine
    from sqlalchemy import text

    engine = get_engine()
    datasets: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in symbols:
        data: dict[str, pd.DataFrame] = {}
        for tf in ("1m", "1h", "4h"):
            t0 = time.monotonic()
            where = "symbol = :sym AND timeframe = :tf"
            params: dict = {"sym": symbol, "tf": tf}
            if exchange:
                where += " AND exchange = :ex"
                params["ex"] = exchange

            sql = text(f"""
                SELECT time, open::double precision, high::double precision,
                       low::double precision, close::double precision,
                       volume::double precision
                FROM candles
                WHERE {where}
                ORDER BY time ASC
            """)

            async with engine.connect() as conn:
                result = await conn.execute(sql, params)
                rows = result.fetchall()

            if not rows:
                log.warning("no_data", symbol=symbol, timeframe=tf)
                continue

            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df = df.set_index("time")
            elapsed = time.monotonic() - t0
            data[tf] = df
            log.info("data_loaded", symbol=symbol, timeframe=tf, rows=len(df),
                     start=str(df.index[0]), end=str(df.index[-1]),
                     elapsed=f"{elapsed:.2f}s")

        if data:
            datasets[symbol] = data

    return datasets


def train(args: argparse.Namespace, datasets: dict[str, dict[str, pd.DataFrame]]) -> None:
    from smart_trader.agent.meta_controller import MetaController
    from smart_trader.agent.trainer import PPOConfig, PPOTrainer
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.market_env import MarketEnv, MarketEnvConfig
    from smart_trader.env.reward import RewardConfig
    from smart_trader.env.spaces import SpaceConfig

    first_sym = list(datasets.keys())[0]
    first_data = datasets[first_sym]

    sample_tf = list(first_data.keys())[0]
    sample_feats = compute_features(first_data[sample_tf].head(50), FeatureConfig(), prefix="x_")
    actual_features_per_tf = len(sample_feats.columns)
    log.info("feature_dim_detected", features_per_tf=actual_features_per_tf,
             columns=list(sample_feats.columns)[:5])

    n_tf = len(first_data)
    space_cfg = SpaceConfig(
        n_timeframes=n_tf,
        features_per_tf=actual_features_per_tf,
        lookback=args.lookback,
    )

    primary_tf = "1m" if "1m" in first_data else list(first_data.keys())[0]
    min_bars = min(len(d[primary_tf]) for d in datasets.values() if primary_tf in d)
    max_episode = min(args.max_episode, min_bars // 2)
    log.info("episode_config", symbols=list(datasets.keys()), primary_tf=primary_tf,
             min_bars=min_bars, max_episode=max_episode)

    env_config = MarketEnvConfig(
        datasets=datasets,
        primary_tf=primary_tf,
        max_episode_bars=max_episode,
        initial_cash=10_000.0,
        space_config=space_cfg,
        reward_config=RewardConfig(),
    )

    env = MarketEnv(env_config)
    obs_dim = env.observation_space.shape[0]
    log.info("env_created", obs_dim=obs_dim, action_space=str(env.action_space))

    # quick sanity: reset + step
    obs, info = env.reset()
    assert obs.shape == (obs_dim,), f"obs shape mismatch: {obs.shape} vs ({obs_dim},)"
    assert not np.any(np.isnan(obs)), "NaN in initial observation"

    test_action = env.action_space.sample()
    obs2, reward, term, trunc, info2 = env.step(test_action)
    assert obs2.shape == (obs_dim,), f"step obs shape: {obs2.shape}"
    log.info("sanity_check_passed", reward=f"{reward:.6f}", info_keys=list(info2.keys()))

    # build agent
    agent = MetaController(
        obs_dim=obs_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        lookback=space_cfg.lookback,
        context_dim=space_cfg.context_dim,
    )

    param_count = sum(p.numel() for p in agent.network.parameters())
    log.info("agent_created", params=f"{param_count:,}", d_model=args.d_model,
             n_layers=args.n_layers, device=args.device)

    # train
    ppo_config = PPOConfig(
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        entropy_coef=0.02,
        eval_freq=5,
        eval_episodes=5,
        patience=10,
    )

    trainer = PPOTrainer(env, agent, ppo_config)

    resume_step = 0
    if args.resume:
        meta = agent.load(args.resume, optimizer=trainer.optimizer)
        resume_step = meta.get("step", 0)
        rn_state = meta.get("reward_norm")
        if rn_state:
            trainer._reward_norm.load_state_dict(rn_state)
        log.info("resumed_from_checkpoint", path=args.resume, step=resume_step)

    sym_str = ", ".join(datasets.keys())
    print(f"\n{'═'*60}")
    print(f"  PPO Training — {args.timesteps:,} steps" +
          (f" (resume from {resume_step:,})" if resume_step else ""))
    print(f"{'═'*60}")
    print(f"  symbols:    {sym_str}")
    print(f"  obs_dim:    {obs_dim}")
    print(f"  lookback:   {args.lookback}")
    print(f"  d_model:    {args.d_model}")
    print(f"  params:     {param_count:,}")
    print(f"  n_steps:    {args.n_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_episode:{max_episode}")
    print(f"{'═'*60}\n")

    t0 = time.monotonic()
    stats = trainer.train(
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=5,
        resume_step=resume_step,
    )
    elapsed = time.monotonic() - t0

    # save final
    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "final_agent.pt"
    agent.save(save_path)

    # report
    updates = stats.get("updates", [])
    print(f"\n{'═'*60}")
    status = "Early Stopped" if stats.get("stopped_early") else "Complete"
    print(f"  Training {status} — {elapsed:.1f}s")
    print(f"{'═'*60}")
    print(f"  Total updates:   {stats.get('total_updates', 0)}")
    print(f"  Mean reward:     {stats.get('mean_reward', 0):.4f}")
    print(f"  Std reward:      {stats.get('std_reward', 0):.4f}")
    print(f"  Best eval reward:{stats.get('best_eval_reward', 0):.4f}")
    print(f"  Saved to:        {save_path}")

    if updates:
        print(f"\n  {'Update':>6} {'Reward':>10} {'P.Loss':>10} {'V.Loss':>10} {'Entropy':>10}")
        print(f"  {'─'*50}")
        for u in updates:
            print(f"  {u.get('total_steps', 0):>6,} "
                  f"{u.get('rollout_reward', 0):>10.4f} "
                  f"{u.get('policy_loss', 0):>10.4f} "
                  f"{u.get('value_loss', 0):>10.4f} "
                  f"{u.get('entropy', 0):>10.4f}")

    print(f"{'═'*60}\n")


def main() -> None:
    args = parse_args()
    datasets = asyncio.run(load_data(args.symbols, exchange=args.exchange))

    if not datasets:
        log.error("no_data_available", hint="Run: uv run python scripts/backfill_data.py first")
        return

    if args.per_symbol:
        base_dir = args.checkpoint_dir
        for sym in datasets:
            safe = sym.replace("/", "_")
            args.checkpoint_dir = f"{base_dir}/{safe}"
            log.info("per_symbol_training", symbol=sym)
            train(args, {sym: datasets[sym]})
    else:
        train(args, datasets)


if __name__ == "__main__":
    main()
