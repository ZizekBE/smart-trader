"""Walk-forward out-of-sample evaluation of a trained RL agent.

Splits 2-year BTC data into rolling windows:
  - Train window is skipped (agent already trained)
  - Test on N consecutive non-overlapping 7-day windows

The agent runs in deterministic mode — no exploration.

Usage::

    uv run python scripts/eval_walkforward.py \
        --checkpoint ./checkpoints/binance_2m/checkpoint_635.pt

Architecture flags default to the same values as ``train_rl_agent.py``.
Checkpoints saved after this change embed ``n_layers`` / ``n_heads`` in
``config``; older files fall back to counting ``encoder.layers.*`` keys.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog  # noqa: E402

from smart_trader.env.sim_profiles import build_sim_config, sim_config_to_meta  # noqa: E402
from smart_trader.env.simulator import SimulatorConfig  # noqa: E402

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward RL agent evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to agent .pt checkpoint")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                    help="One or more symbols to evaluate")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--test-days", type=int, default=7, help="Test window in days")
    p.add_argument("--n-folds", type=int, default=20, help="Number of test folds per symbol")
    p.add_argument(
        "--d-model", type=int, default=64,
        help="Transformer width (must match checkpoint; default matches train_rl_agent.py)",
    )
    p.add_argument(
        "--n-layers", type=int, default=1,
        help="Transformer depth (must match checkpoint; default matches train_rl_agent.py)",
    )
    p.add_argument(
        "--n-heads", type=int, default=4,
        help="Attention heads (must match checkpoint; default matches train_rl_agent.py)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output", default=None,
        help="Path to save results (CSV if .csv, JSON otherwise). "
             "Auto-generated next to checkpoint when omitted.",
    )
    p.add_argument(
        "--cost-profile",
        choices=("default", "conservative"),
        default="default",
        help="Simulator fees/slippage/depth for WF folds (conservative = higher friction).",
    )
    return p.parse_args()


async def load_all_data(
    symbols: list[str], exchange: str, load_funding_rates: bool = False,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.Series]]:
    """Load candle data (and optionally funding rates) for multiple symbols."""
    from sqlalchemy import text

    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    datasets: dict[str, dict[str, pd.DataFrame]] = {}
    funding_rates: dict[str, pd.Series] = {}

    for symbol in symbols:
        data: dict[str, pd.DataFrame] = {}
        for tf in ("1m", "1h", "4h"):
            sql = text("""
                SELECT time, open::double precision, high::double precision,
                       low::double precision, close::double precision,
                       volume::double precision
                FROM candles
                WHERE exchange = :ex AND symbol = :sym AND timeframe = :tf
                ORDER BY time ASC
            """)
            async with engine.connect() as conn:
                result = await conn.execute(sql, {"ex": exchange, "sym": symbol, "tf": tf})
                rows = result.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
                df = df.set_index("time")
                data[tf] = df
                log.info("loaded", symbol=symbol, tf=tf, rows=len(df))
        if data:
            datasets[symbol] = data

        if load_funding_rates:
            fr_sql = text("""
                SELECT time, rate::double precision
                FROM funding_rates
                WHERE exchange = :ex AND symbol = :sym
                ORDER BY time ASC
            """)
            async with engine.connect() as conn:
                fr_rows = await conn.execute(fr_sql, {"ex": exchange, "sym": symbol})
                fr_data = fr_rows.fetchall()
            if fr_data:
                fr_df = pd.DataFrame(fr_data, columns=["time", "rate"])
                fr_df = fr_df.set_index("time")
                funding_rates[symbol] = fr_df["rate"].sort_index()
                log.info("funding_rates_loaded", symbol=symbol, rows=len(fr_df))

    return datasets, funding_rates


_LAYER_RE = re.compile(r"^backbone\.encoder\.layers\.(\d+)\.")


def _infer_n_layers(state: dict) -> int | None:
    idx: list[int] = []
    for k in state:
        m = _LAYER_RE.match(k)
        if m:
            idx.append(int(m.group(1)))
    return (max(idx) + 1) if idx else None


def resolve_arch_from_checkpoint(
    path: Path, device: str, args: argparse.Namespace,
) -> dict:
    """Read architecture params from checkpoint config or state_dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    state = ckpt.get("model_state") or {}

    d_model = int(cfg.get("d_model", args.d_model))
    n_layers = cfg.get("n_layers")
    n_layers = int(n_layers) if n_layers is not None else _infer_n_layers(state)
    if n_layers is None:
        n_layers = args.n_layers

    n_heads = int(cfg.get("n_heads", args.n_heads))
    lookback = int(cfg.get("lookback", 1))
    context_dim = int(cfg.get("context_dim", 0))

    saved_obs = cfg.get("obs_dim")
    if saved_obs is not None:
        saved_obs = int(saved_obs)
    log.info(
        "checkpoint_arch",
        path=str(path),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        lookback=lookback,
        obs_dim_saved=saved_obs,
    )
    return {
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "lookback": lookback, "context_dim": context_dim,
        "saved_obs": saved_obs,
    }


def evaluate_fold(
    agent,
    data_slice: dict[str, pd.DataFrame],
    fold: int,
    obs_dim: int,
    lookback: int = 1,
    *,
    sim_config: SimulatorConfig | None = None,
    microstructure_dim: int = 0,
    funding_rates: dict[str, pd.Series] | None = None,
) -> dict:
    """Run the agent in deterministic mode on a data slice."""
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.market_env import MarketEnv, MarketEnvConfig
    from smart_trader.env.reward import RewardConfig
    from smart_trader.env.spaces import SpaceConfig

    sample_tf = list(data_slice.keys())[0]
    sample_feats = compute_features(data_slice[sample_tf].head(50), FeatureConfig(), prefix="x_")
    features_per_tf = len(sample_feats.columns)

    space_cfg = SpaceConfig(
        n_timeframes=len(data_slice),
        features_per_tf=features_per_tf,
        lookback=lookback,
        microstructure_dim=microstructure_dim,
    )
    primary_tf = "1m" if "1m" in data_slice else list(data_slice.keys())[0]
    max_bars = len(data_slice[primary_tf]) - 1

    sc = sim_config if sim_config is not None else SimulatorConfig()
    env_config = MarketEnvConfig(
        data=data_slice,
        primary_tf=primary_tf,
        max_episode_bars=max_bars,
        initial_cash=10_000.0,
        space_config=space_cfg,
        reward_config=RewardConfig(),
        sim_config=sc,
        funding_rates=funding_rates or None,
    )

    env = MarketEnv(env_config)
    agent.set_deterministic(True)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _, _ = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return {
        "fold": fold,
        "steps": steps,
        "total_reward": total_reward,
        **info,
    }


def _run_symbol_folds(
    agent,
    data: dict[str, pd.DataFrame],
    symbol: str,
    args,
    *,
    verbose: bool = True,
    lookback: int = 1,
    sim_config: SimulatorConfig | None = None,
    microstructure_dim: int = 0,
    funding_rates: dict[str, pd.Series] | None = None,
) -> list[dict]:
    """Walk-forward folds for a single symbol."""
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.spaces import SpaceConfig

    sample_feats = compute_features(data["1m"].head(50), FeatureConfig(), prefix="x_")
    features_per_tf = len(sample_feats.columns)
    space_cfg = SpaceConfig(n_timeframes=len(data), features_per_tf=features_per_tf,
                            lookback=lookback, microstructure_dim=microstructure_dim)
    obs_dim = space_cfg.lookback * space_cfg.market_dim + space_cfg.context_dim

    bars_per_day = 24 * 60
    test_bars = args.test_days * bars_per_day
    total_1m = len(data["1m"])
    results = []

    for fold_idx in range(args.n_folds):
        end_idx = total_1m - fold_idx * test_bars
        start_idx = end_idx - test_bars
        if start_idx < 200:
            break

        fold_data: dict[str, pd.DataFrame] = {}
        t_start_fold: pd.Timestamp | None = None
        t_end_fold: pd.Timestamp | None = None
        for tf, df in data.items():
            if tf == "1m":
                fold_data[tf] = df.iloc[start_idx:end_idx]
                t_start_fold = df.index[start_idx]
                t_end_fold = df.index[min(end_idx, len(df) - 1)]
            else:
                t_start = data["1m"].index[start_idx]
                t_end = data["1m"].index[min(end_idx, len(data["1m"]) - 1)]
                mask = (df.index >= t_start) & (df.index <= t_end)
                fold_data[tf] = df.loc[mask]

        if len(fold_data.get("1m", pd.DataFrame())) < 1000:
            break

        fold_fr: dict[str, pd.Series] | None = None
        if funding_rates and t_start_fold is not None and t_end_fold is not None:
            fold_fr = {}
            for fr_sym, fr_series in funding_rates.items():
                mask = (fr_series.index >= t_start_fold) & (fr_series.index <= t_end_fold)
                fold_fr[fr_sym] = fr_series.loc[mask]

        t0 = time.monotonic()
        result = evaluate_fold(
            agent, fold_data, fold_idx, obs_dim, lookback=lookback, sim_config=sim_config,
            microstructure_dim=microstructure_dim, funding_rates=fold_fr,
        )
        result["elapsed"] = time.monotonic() - t0
        result["symbol"] = symbol
        results.append(result)

        period_start = fold_data["1m"].index[0].strftime("%m/%d")
        period_end = fold_data["1m"].index[-1].strftime("%m/%d")
        period = f"{period_start} - {period_end}"
        tr = result.get("total_return", 0)
        sh = result.get("sharpe", 0)
        md = result.get("max_dd", 0)
        nt = result.get("n_trades", 0)
        marker = "+" if tr > 0 else " "
        if verbose:
            print(f"  {fold_idx+1:>4}  {symbol:>10}  {period:>16}  "
                  f"{marker}{tr:>7.2%}  {sh:>8.2f}  {md:>7.2%}  {nt:>6}")

    return results


def summarize_fold_results(
    all_results: list[dict], n_symbols: int,
) -> dict[str, float | int]:
    """Aggregate walk-forward metrics (same definitions as printed summary)."""
    if not all_results:
        return {}
    returns = [float(r.get("total_return", 0)) for r in all_results]
    sharpes = [float(r.get("sharpe", 0)) for r in all_results]
    max_dds = [float(r.get("max_dd", 0)) for r in all_results]
    wins = sum(1 for r in returns if r > 0)
    return {
        "n_folds": len(all_results),
        "n_symbols": n_symbols,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "win_rate": wins / len(returns),
        "wins": wins,
        "mean_sharpe": float(np.mean(sharpes)),
        "mean_max_dd": float(np.mean(max_dds)),
        "best_fold": float(max(returns)),
        "worst_fold": float(min(returns)),
    }


def save_eval_results(
    all_results: list[dict],
    summary: dict,
    output_path: Path,
    meta: dict | None = None,
) -> None:
    """Persist fold-level results and aggregate summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
    else:
        payload: dict = {"summary": summary, "folds": all_results}
        if meta:
            payload["meta"] = meta
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    log.info("results_saved", path=str(output_path))


def run_walkforward_eval(
    agent,
    all_data: dict[str, dict[str, pd.DataFrame]],
    args,
    *,
    verbose: bool = True,
    lookback: int = 1,
    sim_config: SimulatorConfig | None = None,
    microstructure_dim: int = 0,
    funding_rates: dict[str, pd.Series] | None = None,
) -> list[dict]:
    """Run all symbols × folds; optionally suppress per-fold lines (for sweeps)."""
    all_results: list[dict] = []
    for sym, sym_data in all_data.items():
        results = _run_symbol_folds(
            agent, sym_data, sym, args,
            verbose=verbose, lookback=lookback, sim_config=sim_config,
            microstructure_dim=microstructure_dim, funding_rates=funding_rates,
        )
        all_results.extend(results)
    return all_results


def main() -> None:
    args = parse_args()

    from smart_trader.agent.meta_controller import MetaController
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.spaces import SpaceConfig

    # Probe checkpoint to detect if microstructure was used during training
    ckpt_path_probe = Path(args.checkpoint)
    raw_ckpt = torch.load(ckpt_path_probe, map_location="cpu", weights_only=False)
    saved_obs_probe = int((raw_ckpt.get("config") or {}).get("obs_dim", 0))
    need_fr = saved_obs_probe > 0  # will refine after computing base obs_dim

    all_data, funding_rates = asyncio.run(
        load_all_data(args.symbols, args.exchange, load_funding_rates=need_fr)
    )
    all_data = {sym: d for sym, d in all_data.items() if "1m" in d}

    if not all_data:
        log.error("no_data_available")
        return

    ckpt_path = Path(args.checkpoint)
    arch = resolve_arch_from_checkpoint(ckpt_path, args.device, args)
    lookback = arch["lookback"]
    context_dim = arch["context_dim"]

    first_data = list(all_data.values())[0]
    sample_feats = compute_features(first_data["1m"].head(50), FeatureConfig(), prefix="x_")
    features_per_tf = len(sample_feats.columns)
    # Compute base obs_dim (no microstructure) then detect extra dims from checkpoint
    space_cfg_base = SpaceConfig(n_timeframes=len(first_data), features_per_tf=features_per_tf,
                                 lookback=lookback)
    base_obs_dim = space_cfg_base.lookback * space_cfg_base.market_dim + space_cfg_base.context_dim

    saved_obs = arch["saved_obs"]
    microstructure_dim = max(0, (saved_obs - base_obs_dim)) if saved_obs is not None else 0
    obs_dim = base_obs_dim + microstructure_dim

    if microstructure_dim > 0:
        log.info("microstructure_detected", microstructure_dim=microstructure_dim,
                 obs_dim=obs_dim, fr_symbols=list(funding_rates.keys()))
    elif saved_obs is not None and saved_obs != base_obs_dim:
        log.warning("obs_dim_mismatch", from_data=base_obs_dim, from_checkpoint=saved_obs)

    agent = MetaController(
        obs_dim=obs_dim,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        device=args.device,
        lookback=lookback,
        context_dim=context_dim,
    )
    try:
        agent.load(ckpt_path)
    except RuntimeError as e:
        if "state_dict" in str(e) or "size mismatch" in str(e) or "Missing key" in str(e):
            print(
                "\n  Failed to load checkpoint — model shape does not match.\n"
                "  Try explicit flags to match training, e.g.:\n"
                "    --d-model 64 --n-layers 1 --n-heads 4\n"
                "  (defaults now match train_rl_agent.py.)\n",
                file=sys.stderr,
            )
        raise
    log.info("agent_loaded", checkpoint=args.checkpoint, obs_dim=obs_dim,
             symbols=list(all_data.keys()))

    sim_config = build_sim_config(args.cost_profile)
    sim_meta = sim_config_to_meta(sim_config)
    log.info("wf_simulator", cost_profile=args.cost_profile, **sim_meta)

    print(f"\n{'═'*80}")
    n_sym = len(all_data)
    print(f"  Walk-Forward Evaluation — {n_sym} symbols × {args.n_folds} folds × {args.test_days}d")
    dp = float(sim_config.depth_profile_usd)
    print(
        f"  Cost profile: {args.cost_profile}  |  taker_fee={sim_config.taker_fee:.5f}  "
        f"maker_fee={sim_config.maker_fee:.5f}  slippage_bps={sim_config.slippage_bps}  "
        f"depth_usd={dp:,.0f}",
    )
    print(f"{'═'*80}")
    print(f"  {'Fold':>4}  {'Symbol':>10}  {'Period':>16}  "
          f"{'Return':>8}  {'Sharpe':>8}  {'MaxDD':>8}  {'Trades':>6}")
    print(f"  {'─'*72}")

    all_results = run_walkforward_eval(
        agent, all_data, args, verbose=True, lookback=lookback, sim_config=sim_config,
        microstructure_dim=microstructure_dim,
        funding_rates=funding_rates if microstructure_dim > 0 else None,
    )

    print(f"  {'─'*72}")

    if all_results:
        agg = summarize_fold_results(all_results, len(all_data))
        print(f"\n  Overall ({agg['n_folds']} folds across {len(all_data)} symbols):")
        print(f"    Mean return:   {agg['mean_return']:>+.2%}")
        print(f"    Std return:    {agg['std_return']:>.2%}")
        print(f"    Win rate:      {agg['win_rate']:.0%} ({agg['wins']}/{agg['n_folds']})")
        print(f"    Mean Sharpe:   {agg['mean_sharpe']:.2f}")
        print(f"    Mean MaxDD:    {agg['mean_max_dd']:.2%}")
        print(f"    Best fold:     {agg['best_fold']:>+.2%}")
        print(f"    Worst fold:    {agg['worst_fold']:>+.2%}")

        for sym in all_data:
            sym_res = [r for r in all_results if r.get("symbol") == sym]
            if not sym_res:
                continue
            sr = [r.get("total_return", 0) for r in sym_res]
            wr = sum(1 for r in sr if r > 0) / len(sr)
            print(f"\n  {sym} ({len(sym_res)} folds):")
            print(f"    Mean return: {np.mean(sr):>+.2%}  Win rate: {wr:.0%}  "
                  f"Mean MaxDD: {np.mean([r.get('max_dd', 0) for r in sym_res]):.2%}")

        out_path = (
            ckpt_path.with_suffix(".eval.json")
            if args.output is None
            else Path(args.output)
        )
        wf_meta = {
            "checkpoint": str(ckpt_path.resolve()),
            "symbols": list(args.symbols),
            "exchange": args.exchange,
            "n_folds": args.n_folds,
            "test_days": args.test_days,
            "cost_profile": args.cost_profile,
            "simulator": sim_meta,
        }
        save_eval_results(all_results, agg, out_path, meta=wf_meta)

    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()
