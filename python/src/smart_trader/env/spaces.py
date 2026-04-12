"""Observation and action space builders for the MarketEnv.

Defines the structured observation (multi-TF features + portfolio state +
market microstructure + time embeddings) and hierarchical action space
(regime belief + target position + risk budget + hold duration).
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SpaceConfig:
    n_timeframes: int = 4        # e.g. 1m, 5m, 1h, 4h
    features_per_tf: int = 30   # indicators per timeframe
    portfolio_dim: int = 6       # position, pnl, margin, cash, leverage, dd
    microstructure_dim: int = 0  # reserved; set >0 when real microstructure data is available
    time_dim: int = 4            # hour_sin, hour_cos, dow_sin, dow_cos


def build_observation_space(cfg: SpaceConfig | None = None) -> spaces.Box:
    """Flat observation vector for the Meta Agent.

    Layout:
      [multi_tf_features | portfolio_state | microstructure | time_embedding]
    """
    c = cfg or SpaceConfig()
    total = (
        c.n_timeframes * c.features_per_tf
        + c.portfolio_dim
        + c.microstructure_dim
        + c.time_dim
    )
    return spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)


def build_action_space() -> spaces.Dict:
    """Hierarchical action space for the Meta Agent.

    Components:
      - position:   Box([-1, 1]) — target position (-1 full short to +1 full long)
      - risk_budget: Box([0.01, 0.10]) — max portfolio risk % for this decision
      - hold_bars:  Discrete(3) — hold for 15m(0) / 1h(1) / 4h(2)
    """
    return spaces.Dict({
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "risk_budget": spaces.Box(low=0.01, high=0.10, shape=(1,), dtype=np.float32),
        "hold_bars": spaces.Discrete(3),
    })


def flatten_action(action: dict) -> np.ndarray:
    """Convert Dict action to a flat array for network output."""
    return np.concatenate([
        np.atleast_1d(action["position"]).astype(np.float32),
        np.atleast_1d(action["risk_budget"]).astype(np.float32),
        np.array([action["hold_bars"]], dtype=np.float32),
    ])


def unflatten_action(flat: np.ndarray) -> dict:
    """Inverse of flatten_action."""
    return {
        "position": np.clip(flat[0:1], -1, 1).astype(np.float32),
        "risk_budget": np.clip(flat[1:2], 0.01, 0.10).astype(np.float32),
        "hold_bars": int(np.clip(np.round(flat[2]), 0, 2)),
    }
