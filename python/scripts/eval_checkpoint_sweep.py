"""Compare walk-forward metrics across multiple RL checkpoints (early stopping probe).

Loads DB data once, then evaluates each checkpoint with the same folds as
``eval_walkforward.py`` (quiet per-fold output, summary table only).

Usage::

    cd python && uv run python scripts/eval_checkpoint_sweep.py \\
        --checkpoint-dir ./checkpoints/rl_500k \\
        --symbols BTC/USDT ETH/USDT --exchange binance

    # Only specific update indices (files must exist):
    uv run python scripts/eval_checkpoint_sweep.py \\
        --checkpoint-dir ./checkpoints/rl_500k --pick 200 400 600 975
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_eval_walkforward():
    path = ROOT / "scripts" / "eval_walkforward.py"
    spec = importlib.util.spec_from_file_location("eval_walkforward", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load eval_walkforward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep walk-forward across checkpoints")
    p.add_argument("--checkpoint-dir", required=True, type=Path, help="Directory with checkpoint_*.pt")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--n-folds", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Evaluate checkpoint_N.pt when N%%stride==0, plus first/last numbered; "
        "0 = every numbered file (can be slow). Default 50 for a denser grid than legacy 200.",
    )
    p.add_argument(
        "--grid",
        choices=("dense", "full"),
        default=None,
        help="dense: stride=25; full: all checkpoint_*.pt (overrides --stride).",
    )
    p.add_argument(
        "--pick",
        type=int,
        nargs="*",
        default=None,
        help="If set, only these update indices (overrides --stride and --grid)",
    )
    p.add_argument(
        "--no-final",
        action="store_true",
        help="Skip final_agent.pt",
    )
    ns = p.parse_args()
    if ns.pick is None and ns.grid == "dense":
        ns.stride = 25
    elif ns.pick is None and ns.grid == "full":
        ns.stride = 0
    return ns


def discover_numbered_ckpts(d: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted(d.glob("checkpoint_*.pt")):
        name = p.stem
        if not name.startswith("checkpoint_"):
            continue
        try:
            n = int(name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((n, p))
    out.sort(key=lambda x: x[0])
    return out


def select_paths(
    d: Path,
    stride: int,
    pick: list[int] | None,
    include_final: bool,
) -> list[tuple[str, Path]]:
    """Return (label, path) in evaluation order."""
    numbered = discover_numbered_ckpts(d)
    if not numbered and not (include_final and (d / "final_agent.pt").exists()):
        return []

    chosen: list[tuple[str, Path]] = []

    if pick is not None:
        by_n = {n: p for n, p in numbered}
        for n in sorted(pick):
            if n in by_n:
                chosen.append((f"checkpoint_{n}", by_n[n]))
            else:
                print(f"  [skip] checkpoint_{n}.pt not found in {d}", file=sys.stderr)
    else:
        if numbered:
            ns = [n for n, _ in numbered]
            first_n, last_n = ns[0], ns[-1]
            for n, p in numbered:
                if stride <= 0:
                    chosen.append((f"checkpoint_{n}", p))
                elif n == first_n or n == last_n or n % stride == 0:
                    chosen.append((f"checkpoint_{n}", p))
        # de-dup while preserving order
        seen: set[Path] = set()
        deduped: list[tuple[str, Path]] = []
        for lab, p in chosen:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                deduped.append((lab, p))
        chosen = deduped

    if include_final:
        fin = d / "final_agent.pt"
        if fin.exists():
            chosen.append(("final_agent", fin))

    return chosen


def main() -> None:
    args = parse_args()
    ew = _load_eval_walkforward()

    from smart_trader.agent.meta_controller import MetaController
    from smart_trader.data.features.engine import FeatureConfig, compute_features
    from smart_trader.env.spaces import SpaceConfig

    d = args.checkpoint_dir.resolve()
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        sys.exit(1)

    items = select_paths(d, args.stride, args.pick, include_final=not args.no_final)
    if not items:
        print(f"No checkpoints found under {d}", file=sys.stderr)
        sys.exit(1)

    mode = "pick" if args.pick is not None else f"stride={args.stride}"
    if args.grid and args.pick is None:
        mode = f"{args.grid}({mode})"
    print(f"  sweep: {len(items)} checkpoints, mode={mode}, final={'no' if args.no_final else 'yes'}")

    all_data = asyncio.run(ew.load_all_data(args.symbols, args.exchange))
    all_data = {sym: x for sym, x in all_data.items() if "1m" in x}
    if not all_data:
        print("no_data_available", file=sys.stderr)
        sys.exit(1)

    first_data = list(all_data.values())[0]
    sample_feats = compute_features(first_data["1m"].head(50), FeatureConfig(), prefix="x_")
    features_per_tf = len(sample_feats.columns)

    eval_args = Namespace(
        test_days=args.test_days,
        n_folds=args.n_folds,
        symbols=args.symbols,
        exchange=args.exchange,
    )

    arch_ns = Namespace(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    rows: list[tuple[str, dict]] = []

    print(f"\n{'═'*100}")
    print(f"  Checkpoint sweep — {len(items)} files, {len(all_data)} symbols × {args.n_folds} folds × {args.test_days}d")
    print(f"{'═'*100}")
    hdr = (
        f"  {'Label':<18} {'MeanRet':>10} {'Sharpe':>8} {'Win%':>7} "
        f"{'MeanDD':>8} {'Worst':>9} {'Folds':>6}"
    )
    print(hdr)
    print(f"  {'─'*96}")

    for label, ckpt_path in items:
        arch = ew.resolve_arch_from_checkpoint(ckpt_path, args.device, arch_ns)
        lookback = arch["lookback"]
        context_dim = arch["context_dim"]

        space_cfg = SpaceConfig(n_timeframes=len(first_data),
                                features_per_tf=features_per_tf,
                                lookback=lookback)
        obs_dim = space_cfg.lookback * space_cfg.market_dim + space_cfg.context_dim

        saved_obs = arch["saved_obs"]
        if saved_obs is not None and saved_obs != obs_dim:
            print(f"  {label:<18}  SKIP obs_dim ckpt={saved_obs} data={obs_dim}", file=sys.stderr)
            continue

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
            print(f"  {label:<18}  LOAD FAILED: {e}", file=sys.stderr)
            continue

        all_results = ew.run_walkforward_eval(agent, all_data, eval_args,
                                              verbose=False, lookback=lookback)
        agg = ew.summarize_fold_results(all_results, len(all_data))
        if not agg:
            print(f"  {label:<18}  no results")
            continue

        rows.append((label, agg))
        print(
            f"  {label:<18} {agg['mean_return']:>+9.2%} {agg['mean_sharpe']:>8.2f} "
            f"{agg['win_rate']:>6.0%} {agg['mean_max_dd']:>7.2%} {agg['worst_fold']:>+8.2%} "
            f"{agg['n_folds']:>6}"
        )

    print(f"  {'─'*96}")

    if rows:
        best = max(rows, key=lambda x: x[1]["mean_sharpe"])
        also = max(rows, key=lambda x: x[1]["mean_return"])
        print(f"\n  Best mean Sharpe:  {best[0]}  (Sharpe={best[1]['mean_sharpe']:.2f}, "
              f"ret={best[1]['mean_return']:+.2%})")
        print(f"  Best mean return:  {also[0]}  (ret={also[1]['mean_return']:+.2%}, "
              f"Sharpe={also[1]['mean_sharpe']:.2f})")
    print(f"{'═'*100}\n")


if __name__ == "__main__":
    main()
