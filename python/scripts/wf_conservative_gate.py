"""Check a walk-forward JSON against conservative-cost summary gates.

Use after ``eval_walkforward.py --cost-profile conservative ...`` so ``meta``
contains ``cost_profile`` and ``simulator`` settings.

Thresholds live in ``configs/wf_gates.json`` (adjust ``profiles.*``). Default
CLI profile is that file's ``default_profile`` (currently ``shadow``).

Exit 0 if all gates pass, 1 otherwise. Exit 2 if ``require_cost_profile`` is set
but the JSON is missing meta or the wrong profile (re-run WF with conservative).

Example::

    cd python && uv run python scripts/wf_conservative_gate.py \\
        checkpoints/wf_ablation_s42_eth/no_l1/wf_eth_cost_conservative.eval.json

Stricter tier (aspirational stability / return)::

    uv run python scripts/wf_conservative_gate.py wf.eval.json --profile target

ETH churn-hardening retrain (then re-run conservative WF + this gate)::

    uv run python scripts/train_rl_agent.py --symbols ETH/USDT --exchange binance \\
      --trade-penalty 0.025 --patience 40 --timesteps 500000 \\
      --n-steps 2048 --batch-size 256 --max-episode 4000 --seed 42 \\
      --checkpoint-dir ./checkpoints/eth_tp0025_pat40

Shadow (after gate passes)::

    uv run python scripts/run_shadow.py --symbol ETH/USDT --model path/to/best_agent.pt \\
      --days 14 --log-file shadow_eth.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THRESH_KEYS = (
    "min_mean_return",
    "max_mean_max_dd",
    "min_win_rate",
    "min_mean_sharpe",
    "max_std_return",
)
_META_KEYS = frozenset(
    {"schema_version", "tuning", "require_cost_profile", "default_profile", "profiles"},
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Gate WF JSON summary (conservative cost)")
    p.add_argument("wf_json", type=Path, help="Path to eval_walkforward output .json")
    p.add_argument(
        "--gates-file",
        type=Path,
        default=root / "configs" / "wf_gates.json",
        help="JSON: either flat thresholds or { profiles: { name: { thresholds... } } }",
    )
    p.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Named tier under profiles.* (default: default_profile in the JSON file)",
    )
    p.add_argument(
        "--soft",
        action="store_true",
        help="Print failures but always exit 0 (for dashboards)",
    )
    return p.parse_args()


def _load_gates(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("gates file must be a JSON object")
    return raw


def _load_wf(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _threshold_dict(d: dict) -> dict:
    return {k: d[k] for k in _THRESH_KEYS if k in d}


def _resolve_profile_rules(raw: dict, profile: str | None) -> tuple[str, dict, str]:
    """Return (profile_name, threshold_rules, description)."""
    if "profiles" in raw:
        pname = (profile or raw.get("default_profile") or "shadow").strip()
        profs = raw["profiles"]
        if pname not in profs:
            avail = ", ".join(sorted(profs))
            raise KeyError(f"profile {pname!r} not in gates file (have: {avail})")
        block = profs[pname]
        if not isinstance(block, dict):
            raise TypeError(f"profiles.{pname} must be an object")
        desc = str(block.get("description", ""))
        return pname, _threshold_dict(block), desc
    # Legacy: flat threshold file
    pname = "flat"
    desc = str(raw.get("description", ""))
    flat = {k: raw[k] for k in _THRESH_KEYS if k in raw}
    return pname, flat, desc


def main() -> None:
    args = parse_args()
    wf_path = args.wf_json
    if not wf_path.is_file():
        print(f"error: WF file not found: {wf_path}", file=sys.stderr)
        sys.exit(2)

    try:
        raw = _load_gates(args.gates_file)
        pname, gates, prof_desc = _resolve_profile_rules(raw, args.profile)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"error: gates file: {e}", file=sys.stderr)
        sys.exit(2)

    wf = _load_wf(wf_path)
    summary = wf.get("summary") or {}
    meta = wf.get("meta") or {}

    require = raw.get("require_cost_profile")
    if require:
        cp = meta.get("cost_profile")
        if not cp:
            print(
                "error: WF JSON has no meta.cost_profile; re-run eval_walkforward "
                "with --cost-profile conservative so gates are meaningful.",
                file=sys.stderr,
            )
            sys.exit(2)
        if str(cp) != str(require):
            print(
                f"error: meta.cost_profile={cp!r} but gates require {require!r}.",
                file=sys.stderr,
            )
            sys.exit(2)

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    mr = float(summary.get("mean_return", 0.0))
    if "min_mean_return" in gates:
        thr = float(gates["min_mean_return"])
        add("min_mean_return", mr >= thr, f"mean_return={mr:.4f} need >={thr:.4f}")

    mdd = float(summary.get("mean_max_dd", 1.0))
    if "max_mean_max_dd" in gates:
        thr = float(gates["max_mean_max_dd"])
        add("max_mean_max_dd", mdd <= thr, f"mean_max_dd={mdd:.4f} need <={thr:.4f}")

    wr = float(summary.get("win_rate", 0.0))
    if "min_win_rate" in gates:
        thr = float(gates["min_win_rate"])
        add("min_win_rate", wr >= thr, f"win_rate={wr:.3f} need >={thr:.3f}")

    ms = float(summary.get("mean_sharpe", 0.0))
    if "min_mean_sharpe" in gates:
        thr = float(gates["min_mean_sharpe"])
        add("min_mean_sharpe", ms >= thr, f"mean_sharpe={ms:.3f} need >={thr:.3f}")

    st = float(summary.get("std_return", 0.0))
    if "max_std_return" in gates and gates["max_std_return"] is not None:
        thr = float(gates["max_std_return"])
        add("max_std_return", st <= thr, f"std_return={st:.4f} need <={thr:.4f}")

    failed = [c for c in checks if not c[1]]

    print(f"  wf:           {wf_path}")
    print(f"  gates file:   {args.gates_file}")
    print(f"  profile:      {pname}")
    if raw.get("tuning"):
        print(f"  tuning note:  {raw['tuning']}")
    if prof_desc:
        print(f"  profile note: {prof_desc}")
    print()

    for name, ok, detail in checks:
        mark = "ok" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    if failed and not args.soft:
        print(f"\n  wf_conservative_gate: FAILED ({len(failed)} check(s))", file=sys.stderr)
        sys.exit(1)

    print("\n  wf_conservative_gate: PASS")
    if not failed:
        print(
            "  next: optional shadow — "
            "uv run python scripts/run_shadow.py --symbol … --model …/best_agent.pt",
        )


if __name__ == "__main__":
    main()
