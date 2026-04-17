"""Shared execution-cost profiles for RL training and walk-forward evaluation.

``default`` matches ``SimulatorConfig()`` library defaults. ``conservative``
uses higher fees, slippage, and shallower depth so train/WF see the same
friction when you opt in.
"""
from __future__ import annotations

from dataclasses import asdict

from smart_trader.env.simulator import SimulatorConfig


def build_sim_config(cost_profile: str) -> SimulatorConfig:
    if cost_profile == "default":
        return SimulatorConfig()
    if cost_profile == "conservative":
        return SimulatorConfig(
            taker_fee=0.001,
            maker_fee=0.0004,
            slippage_bps=15.0,
            depth_profile_usd=200_000.0,
        )
    raise ValueError(f"unknown cost_profile: {cost_profile!r}")


def sim_config_to_meta(cfg: SimulatorConfig) -> dict:
    """JSON-serialisable snapshot for checkpoint ``meta`` / WF ``meta``."""
    d = asdict(cfg)
    fm = d.get("fill_model")
    if hasattr(fm, "value"):
        d["fill_model"] = fm.value
    else:
        d["fill_model"] = str(fm)
    return d
