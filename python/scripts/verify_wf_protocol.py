#!/usr/bin/env python3
"""T-OOS-02-5 — Walk-forward protocol verification.

Validates that a WF eval JSON produced by eval_walkforward.py conforms to the
frozen OOS protocol: correct structure, cost_profile, symbol list, fold count.

Usage
─────
  uv run python scripts/verify_wf_protocol.py \
      ./checkpoints/v9_conservative_r4/best_agent.eval.json

  # Check a different profile:
  uv run python scripts/verify_wf_protocol.py path/to/wf.json --profile conservative

Exit codes:
  0 — all checks pass
  1 — one or more checks failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Frozen protocol values (T-OOS-02-1 through T-OOS-02-4) ───────────────────
FROZEN = {
    "cost_profile":  "conservative",
    "n_folds":       20,
    "test_days":     7,
    "symbols":       ["ETH/USDT"],  # minimum required; BTC/USDT optional
    "exchange":      "binance",
}

REQUIRED_SUMMARY_KEYS = {
    "n_folds", "n_symbols", "mean_return", "std_return",
    "win_rate", "wins", "mean_sharpe", "mean_max_dd",
}
REQUIRED_FOLD_KEYS = {
    "fold", "symbol", "total_return", "sharpe", "max_dd", "n_trades",
}
REQUIRED_META_KEYS = {
    "cost_profile", "n_folds", "test_days", "symbols", "exchange", "simulator",
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def verify(path: Path, expected_profile: str) -> bool:
    print(f"\nVerifying: {path}\n{'─'*60}")

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  ❌  Cannot read JSON: {e}")
        return False

    passes: list[bool] = []

    # ── Top-level structure ───────────────────────────────────────────────────
    print("Structure:")
    passes.append(_check("has 'summary' key",  "summary" in data))
    passes.append(_check("has 'folds' key",    "folds"   in data))
    passes.append(_check("has 'meta' key",     "meta"    in data))

    if "summary" in data:
        missing = REQUIRED_SUMMARY_KEYS - set(data["summary"])
        passes.append(_check("summary keys complete", not missing,
                             f"missing: {missing}" if missing else ""))

    if "folds" in data and data["folds"]:
        fold0 = data["folds"][0]
        missing = REQUIRED_FOLD_KEYS - set(fold0)
        passes.append(_check("fold keys complete", not missing,
                             f"missing: {missing}" if missing else ""))

    if "meta" in data:
        missing = REQUIRED_META_KEYS - set(data["meta"])
        passes.append(_check("meta keys complete", not missing,
                             f"missing: {missing}" if missing else ""))

    # ── Frozen protocol values ────────────────────────────────────────────────
    print("\nFrozen protocol:")
    meta = data.get("meta", {})

    passes.append(_check(
        f"meta.cost_profile == '{expected_profile}'",
        meta.get("cost_profile") == expected_profile,
        f"got: {meta.get('cost_profile')!r}",
    ))
    passes.append(_check(
        f"meta.n_folds == {FROZEN['n_folds']}",
        meta.get("n_folds") == FROZEN["n_folds"],
        f"got: {meta.get('n_folds')}",
    ))
    passes.append(_check(
        f"meta.test_days == {FROZEN['test_days']}",
        meta.get("test_days") == FROZEN["test_days"],
        f"got: {meta.get('test_days')}",
    ))
    passes.append(_check(
        f"meta.exchange == '{FROZEN['exchange']}'",
        meta.get("exchange") == FROZEN["exchange"],
        f"got: {meta.get('exchange')!r}",
    ))

    syms = meta.get("symbols", [])
    required_sym = FROZEN["symbols"][0]
    passes.append(_check(
        f"symbols contains '{required_sym}'",
        required_sym in syms,
        f"got: {syms}",
    ))

    # ── Fold count consistency ────────────────────────────────────────────────
    print("\nConsistency:")
    n_folds_actual = len(data.get("folds", []))
    passes.append(_check(
        f"folds array length == {FROZEN['n_folds']}",
        n_folds_actual == FROZEN["n_folds"],
        f"got: {n_folds_actual}",
    ))
    passes.append(_check(
        "summary.n_folds matches folds array",
        data.get("summary", {}).get("n_folds") == n_folds_actual,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = all(passes)
    total = len(passes)
    passed = sum(passes)
    print(f"\n{'─'*60}")
    print(f"  {'PASS' if ok else 'FAIL'}  {passed}/{total} checks passed")
    if ok:
        s = data.get("summary", {})
        print(f"  mean_sharpe={s.get('mean_sharpe', '?'):.3f}  "
              f"win_rate={s.get('win_rate', '?'):.0%}  "
              f"mean_return={s.get('mean_return', '?'):+.3%}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify WF eval JSON protocol")
    parser.add_argument("eval_json", type=Path, help="Path to *.eval.json file")
    parser.add_argument("--profile", default=FROZEN["cost_profile"],
                        help=f"Expected cost_profile (default: {FROZEN['cost_profile']})")
    args = parser.parse_args()

    ok = verify(args.eval_json, args.profile)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
