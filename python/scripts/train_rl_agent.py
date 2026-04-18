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
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog  # noqa: E402

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
    p.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Transformer hidden dim (smaller = faster)",
    )
    p.add_argument("--n-layers", type=int, default=1, help="Transformer layers")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512, help="PPO rollout length")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-episode", type=int, default=2000, help="Max episode bars")
    p.add_argument("--lookback", type=int, default=10,
                    help="Bars of history in the observation sequence (10 = transformer sequence mode)")
    p.add_argument("--no-regime", action="store_true",
                    help="Disable regime context features (regime_dim=0)")
    p.add_argument("--patience", type=int, default=10,
                    help="Early-stop after this many evals without best_eval improvement")
    p.add_argument("--n-epochs", type=int, default=8,
                    help="PPO update epochs per rollout (was hardcoded 4; higher = better sample efficiency)")
    p.add_argument("--eval-episodes", type=int, default=15,
                    help="Episodes per eval pass (higher = less noise in early stopping signal)")
    p.add_argument("--weight-decay", type=float, default=2e-5,
                    help="AdamW L2 (higher = less OOS variance; default nudged for RL stability)")
    p.add_argument("--trade-penalty", type=float, default=0.02,
                    help="RewardEngine churn penalty per trade (higher = less frequent trades)")
    p.add_argument("--dd-weight", type=float, default=2.0,
                    help="RewardEngine per-step drawdown penalty weight (beta). Default 2.0.")
    p.add_argument("--dd-threshold", type=float, default=0.02,
                    help="Drawdown fraction at which per-step penalty activates. Default 0.02.")
    p.add_argument("--dd-terminal", type=float, default=0.0,
                    help="Terminal episode penalty weight: -(dd_terminal × max_dd × pnl_scale). "
                         "0 = disabled. Use 1.0–2.0 for strong terminal signal.")
    p.add_argument("--sharpe-terminal", type=float, default=0.0,
                    help="Terminal Sharpe reward weight: sharpe_terminal × clip(sharpe/3, -1, 1) × pnl_scale. "
                         "Aligns training with WF Sharpe metric. Use 1.0–2.0.")
    p.add_argument(
        "--cost-profile",
        choices=("default", "conservative"),
        default="default",
        help="MarketEnv simulator fees/slippage; use conservative to match conservative WF.",
    )
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=0.02,
        help="PPO entropy bonus at start of training (linearly decays to --entropy-coef-end).",
    )
    p.add_argument(
        "--entropy-coef-end",
        type=float,
        default=0.004,
        help="Entropy bonus at end of schedule (slightly lower default = less late noise).",
    )
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--exchange", default=None, help="Filter by exchange (binance, gateio)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--per-symbol", action="store_true",
                    help="Train a separate model for each symbol")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for NumPy, PyTorch, and MarketEnv (reproducibility).",
    )
    return p.parse_args()


def _merge_train_meta_into_checkpoints(save_dir: Path, meta: dict) -> None:
    """Persist training frictions into checkpoint ``config`` for traceability."""
    import torch

    for name in ("final_agent.pt", "best_agent.pt"):
        p = save_dir / name
        if not p.is_file():
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ckpt.setdefault("config", {})
        for k, v in meta.items():
            cfg[k] = v
        torch.save(ckpt, p)


def _set_global_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        with contextlib.suppress(AttributeError):
            torch.mps.manual_seed(seed)


async def load_data(
    symbols: list[str], exchange: str | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.Series]]:
    """Load candle data and funding rates for one or more symbols.

    Returns ``(candles, funding_rates)`` where:
      - ``candles``       = ``{symbol: {timeframe: DataFrame}}``
      - ``funding_rates`` = ``{symbol: pd.Series(rate, index=time)}``
    """
    from sqlalchemy import text

    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    datasets: dict[str, dict[str, pd.DataFrame]] = {}
    funding_rates: dict[str, pd.Series] = {}

    for symbol in symbols:
        data: dict[str, pd.DataFrame] = {}
        for tf in ("1m", "1h", "4h", "1d"):
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

        # Load funding rates (8h interval, historical via fetchFundingRateHistory)
        fr_where = "symbol = :sym"
        fr_params: dict = {"sym": symbol}
        if exchange:
            fr_where += " AND exchange = :ex"
            fr_params["ex"] = exchange
        fr_sql = text(f"""
            SELECT time, rate::double precision
            FROM funding_rates
            WHERE {fr_where}
            ORDER BY time ASC
        """)
        async with engine.connect() as conn:
            fr_result = await conn.execute(fr_sql, fr_params)
            fr_rows = fr_result.fetchall()
        if fr_rows:
            fr_df = pd.DataFrame(fr_rows, columns=["time", "rate"]).set_index("time")
            funding_rates[symbol] = fr_df["rate"]
            log.info("funding_rates_loaded", symbol=symbol, rows=len(fr_df),
                     start=str(fr_df.index[0]), end=str(fr_df.index[-1]))
        else:
            log.warning("no_funding_rates", symbol=symbol)

    return datasets, funding_rates


def train(
    args: argparse.Namespace,
    datasets: dict[str, dict[str, pd.DataFrame]],
    funding_rates: dict[str, pd.Series] | None = None,
) -> None:
    from smart_trader.agent.meta_controller import MetaController
    from smart_trader.agent.trainer import PPOConfig, PPOTrainer
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.market_env import MarketEnv, MarketEnvConfig
    from smart_trader.env.reward import RewardConfig
    from smart_trader.env.sim_profiles import build_sim_config, sim_config_to_meta
    from smart_trader.env.spaces import SpaceConfig

    _set_global_seed(args.seed)
    log.info("seed_set", seed=args.seed)

    sim_config = build_sim_config(args.cost_profile)
    log.info("train_simulator", cost_profile=args.cost_profile, **sim_config_to_meta(sim_config))

    first_sym = list(datasets.keys())[0]
    first_data_full = datasets[first_sym]

    # 1d is loaded for regime computation only — exclude from env timeframes
    _ENV_TFS = ("1m", "1h", "4h")
    env_datasets = {
        sym: {tf: df for tf, df in tf_map.items() if tf in _ENV_TFS}
        for sym, tf_map in datasets.items()
    }
    first_data = {tf: df for tf, df in first_data_full.items() if tf in _ENV_TFS}

    sample_tf = list(first_data.keys())[0]
    sample_feats = compute_features(first_data[sample_tf].head(50), FeatureConfig(), prefix="x_")
    actual_features_per_tf = len(sample_feats.columns)
    log.info("feature_dim_detected", features_per_tf=actual_features_per_tf,
             columns=list(sample_feats.columns)[:5])

    n_tf = len(first_data)
    # microstructure_dim = 1 when funding rates are available (FR only; OI deferred until ≥6m history)
    micro_dim = 1 if funding_rates else 0
    if micro_dim:
        covered = [s for s in datasets if s in funding_rates]
        log.info("microstructure_enabled", dim=micro_dim,
                 symbols_with_fr=covered, total_symbols=len(datasets))

    # regime context features (regime_code + vol_code) — pre-computed before env creation
    regime_dim = 0
    regime_features: dict | None = None
    if not getattr(args, "no_regime", False):
        from smart_trader.env.regime_features import compute_regime_features
        regime_features = {}
        regime_dim = 2
        log.info("computing_regime_features", symbols=list(datasets.keys()))
        for sym, tf_map in datasets.items():
            daily_df   = tf_map.get("1d", pd.DataFrame())
            hourly_df  = tf_map.get("1h", pd.DataFrame())
            env_tf_map = env_datasets[sym]
            primary_tf_tmp = "1m" if "1m" in env_tf_map else list(env_tf_map.keys())[0]
            primary_df = env_tf_map[primary_tf_tmp]
            if hourly_df.empty and (daily_df.empty or len(daily_df) < 60):
                log.warning("regime_features: insufficient data for %s — disabling", sym)
                regime_dim = 0
                regime_features = None
                break
            regime_features[sym] = compute_regime_features(
                daily_df, hourly_df, primary_df, sym, stride=1,
            )
        if regime_features:
            log.info("regime_features_ready", symbols=list(regime_features.keys()),
                     regime_dim=regime_dim)

    space_cfg = SpaceConfig(
        n_timeframes=n_tf,
        features_per_tf=actual_features_per_tf,
        lookback=args.lookback,
        microstructure_dim=micro_dim,
        regime_dim=regime_dim,
    )

    primary_tf = "1m" if "1m" in first_data else list(first_data.keys())[0]
    min_bars = min(len(d[primary_tf]) for d in env_datasets.values() if primary_tf in d)
    max_episode = min(args.max_episode, min_bars // 2)
    log.info("episode_config", symbols=list(datasets.keys()), primary_tf=primary_tf,
             min_bars=min_bars, max_episode=max_episode)

    env_config = MarketEnvConfig(
        datasets=env_datasets,
        primary_tf=primary_tf,
        max_episode_bars=max_episode,
        initial_cash=10_000.0,
        space_config=space_cfg,
        reward_config=RewardConfig(
            trade_penalty=args.trade_penalty,
            beta=args.dd_weight,
            dd_threshold=args.dd_threshold,
            dd_terminal_weight=args.dd_terminal,
            sharpe_terminal_weight=args.sharpe_terminal,
        ),
        sim_config=sim_config,
        seed=args.seed,
        funding_rates=funding_rates or None,
        regime_features=regime_features,
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
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        eval_freq=5,
        eval_episodes=args.eval_episodes,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )

    trainer = PPOTrainer(env, agent, ppo_config)

    resume_step = 0
    if args.resume:
        meta = agent.load(args.resume, optimizer=trainer.optimizer)
        resume_step = meta.get("step", 0)
        rn_state = meta.get("reward_norm")
        if rn_state:
            trainer._reward_norm.load_state_dict(rn_state)
        obs_norm_state = meta.get("obs_norm")
        if obs_norm_state and getattr(env, "_obs_norm", None) is not None:
            env._obs_norm.load_state_dict(obs_norm_state)
            log.info("obs_norm_restored", steps_seen=obs_norm_state.get("count", 0))
        log.info("resumed_from_checkpoint", path=args.resume, step=resume_step)

    sym_str = ", ".join(datasets.keys())
    print(f"\n{'═'*60}")
    print(f"  PPO Training — {args.timesteps:,} steps" +
          (f" (resume from {resume_step:,})" if resume_step else ""))
    print(f"{'═'*60}")
    print(f"  symbols:    {sym_str}")
    print(f"  obs_dim:    {obs_dim}")
    print(f"  lookback:   {args.lookback}  regime_dim:{regime_dim}")
    print(f"  d_model:    {args.d_model}")
    print(f"  params:     {param_count:,}")
    print(f"  patience:   {args.patience}")
    print(f"  weight_dec: {args.weight_decay}")
    print(f"  trade_pen:  {args.trade_penalty}")
    print(f"  dd_weight:  {args.dd_weight}  dd_thr:{args.dd_threshold}  dd_term:{args.dd_terminal}  sharpe_term:{args.sharpe_terminal}")
    print(f"  cost_prof:  {args.cost_profile}")
    print(f"  ent_coef:   {args.entropy_coef} -> {args.entropy_coef_end}")
    print(f"  n_steps:    {args.n_steps}")
    print(f"  n_epochs:   {args.n_epochs}")
    print(f"  eval_eps:   {args.eval_episodes}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_episode:{max_episode}")
    print(f"  seed:       {args.seed}")
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

    _merge_train_meta_into_checkpoints(
        save_dir,
        {
            "cost_profile": args.cost_profile,
            "trade_penalty": args.trade_penalty,
            "dd_weight": args.dd_weight,
            "dd_threshold": args.dd_threshold,
            "dd_terminal": args.dd_terminal,
            "train_simulator": sim_config_to_meta(sim_config),
            "weight_decay": args.weight_decay,
            "entropy_coef": args.entropy_coef,
            "entropy_coef_end": args.entropy_coef_end,
            "n_epochs": args.n_epochs,
            "eval_episodes": args.eval_episodes,
        },
    )

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
    datasets, funding_rates = asyncio.run(load_data(args.symbols, exchange=args.exchange))

    if not datasets:
        log.error("no_data_available", hint="Run: uv run python scripts/backfill_data.py first")
        return

    if args.per_symbol:
        base_dir = args.checkpoint_dir
        for sym in datasets:
            safe = sym.replace("/", "_")
            args.checkpoint_dir = f"{base_dir}/{safe}"
            log.info("per_symbol_training", symbol=sym)
            sym_fr = {sym: funding_rates[sym]} if sym in funding_rates else None
            train(args, {sym: datasets[sym]}, sym_fr)
    else:
        train(args, datasets, funding_rates or None)


if __name__ == "__main__":
    main()
