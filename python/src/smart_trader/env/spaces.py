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
    microstructure_dim: int = 0  # set >0 when real microstructure data is available
    time_dim: int = 4            # hour_sin, hour_cos, dow_sin, dow_cos
    regime_dim: int = 0          # regime context: set to 2 for (regime_code, vol_code)
    signal_dim: int = 0          # hybrid mode: 5 = (direction, confidence, is_long, is_short, is_no_signal)
    lookback: int = 1            # bars of history in the observation sequence

    @property
    def market_dim(self) -> int:
        """Feature dimension per bar (all TFs concatenated)."""
        return self.n_timeframes * self.features_per_tf

    @property
    def context_dim(self) -> int:
        """Portfolio + microstructure + time embedding + regime + signal context."""
        return self.portfolio_dim + self.microstructure_dim + self.time_dim + self.regime_dim + self.signal_dim


def build_observation_space(cfg: SpaceConfig | None = None) -> spaces.Box:
    """Flat observation vector for the Meta Agent.

    Layout (lookback=1, legacy):
      [multi_tf_features | portfolio_state | microstructure | time_embedding]

    Layout (lookback>1, sequence):
      [bar_0_features | bar_1_features | ... | bar_{L-1}_features | context]
      where context = portfolio_state | microstructure | time_embedding
    """
    c = cfg or SpaceConfig()
    total = c.lookback * c.market_dim + c.context_dim
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


def build_hybrid_action_space() -> spaces.Box:
    """1D action space for the hybrid position sizer.

    The RL agent outputs a single position_scale ∈ [0, 1] which is
    multiplied by the rule-engine signal direction to get the final position.
    """
    return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


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
