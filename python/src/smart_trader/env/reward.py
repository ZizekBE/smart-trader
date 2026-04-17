"""Reward engine for RL training.

Implements the composite reward function:

    R_t = scaled_pnl (asymmetric) + β·drawdown_penalty
        + γ·cost_penalty + churn_penalty

All components are bounded to prevent extreme values that destabilise
PPO's value function.  An optional RunningNormalizer can wrap the
final scalar reward into ~N(0,1) for stable training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardConfig:
    pnl_scale: float = 40.0          # scale raw step return to useful magnitude
    loss_aversion: float = 1.3       # multiply penalty for negative returns (asymmetric)
    beta: float = 0.5                # drawdown penalty weight
    gamma: float = 0.2               # trading cost penalty weight
    trade_penalty: float = 0.01      # flat penalty per trade (discourages churn)

    dd_threshold: float = 0.05       # drawdown % at which penalty kicks in
    dd_exponent: float = 2.0         # non-linear drawdown penalty exponent

    clip_reward: float = 3.0         # symmetric per-step reward clip


@dataclass
class RewardState:
    """Tracks running statistics needed for reward computation."""
    returns: list[float] = field(default_factory=list)
    peak_value: float = 0.0
    cumulative_costs: float = 0.0
    n_trades: int = 0
    holding_steps: int = 0


class RewardEngine:
    """Computes per-step reward for the RL agent."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()
        self.state = RewardState()

    def reset(self, initial_value: float) -> None:
        self.state = RewardState(peak_value=initial_value)

    def step(
        self,
        portfolio_value: float,
        prev_value: float,
        trade_cost: float = 0.0,
        did_trade: bool = False,
        position_held: bool = False,
    ) -> float:
        """Compute reward for a single environment step.

        Uses scaled PnL with asymmetric loss aversion instead of
        Sortino to avoid instability from dividing by near-zero
        downside deviation.
        """
        step_return = (portfolio_value - prev_value) / (prev_value + 1e-9)
        self.state.returns.append(step_return)

        # --- asymmetric scaled PnL ---
        pnl_reward = step_return * self.cfg.pnl_scale
        if step_return < 0:
            pnl_reward *= self.cfg.loss_aversion

        # --- drawdown penalty (bounded) ---
        self.state.peak_value = max(self.state.peak_value, portfolio_value)
        dd = (self.state.peak_value - portfolio_value) / (self.state.peak_value + 1e-9)
        dd_penalty = 0.0
        if dd > self.cfg.dd_threshold:
            excess = dd - self.cfg.dd_threshold
            dd_penalty = -min(excess ** self.cfg.dd_exponent, 1.0)

        # --- trading cost penalty ---
        self.state.cumulative_costs += trade_cost
        cost_penalty = 0.0
        if did_trade:
            cost_penalty = -trade_cost / (prev_value + 1e-9) * self.cfg.pnl_scale
            cost_penalty = max(cost_penalty, -2.0)
            self.state.n_trades += 1

        if position_held:
            self.state.holding_steps += 1

        churn_penalty = -self.cfg.trade_penalty if did_trade else 0.0

        reward = (
            pnl_reward
            + self.cfg.beta * dd_penalty
            + self.cfg.gamma * cost_penalty
            + churn_penalty
        )
        return float(np.clip(reward, -self.cfg.clip_reward, self.cfg.clip_reward))

    def get_episode_stats(self) -> dict:
        """Summary metrics for the completed episode."""
        returns = np.array(self.state.returns)
        if len(returns) < 2:
            return {"total_return": 0, "sharpe": 0, "sortino": 0, "max_dd": 0}

        total = float(np.prod(1 + returns) - 1)
        mean_r, std_r = float(np.mean(returns)), float(np.std(returns) + 1e-9)
        downside = returns[returns < 0]
        down_std = float(np.std(downside) + 1e-9) if len(downside) > 0 else 1e-9

        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.max((peak - cum) / (peak + 1e-9)))

        return {
            "total_return": total,
            "sharpe": mean_r / std_r * np.sqrt(252 * 24),
            "sortino": mean_r / down_std * np.sqrt(252 * 24),
            "max_dd": max_dd,
            "n_trades": self.state.n_trades,
            "total_costs": self.state.cumulative_costs,
        }


class RunningNormalizer:
    """Welford online reward normalizer — maps rewards to ~N(0,1).

    Used by the PPO trainer to wrap raw env rewards before storing
    them in the rollout buffer.  Tracks exponential moving stats so
    it adapts to reward scale changes over training.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self._gamma = gamma
        self._eps = epsilon
        self._mean = 0.0
        self._var = 1.0
        self._count = 0
        self._ret = 0.0  # discounted return tracker

    def normalize(self, reward: float, done: bool = False) -> float:
        self._ret = reward + self._gamma * self._ret * (1.0 - float(done))
        self._update(self._ret)
        return float(self._ret / (np.sqrt(self._var) + self._eps))

    def state_dict(self) -> dict:
        return {
            "mean": self._mean,
            "var": self._var,
            "count": self._count,
            "ret": self._ret,
        }

    def load_state_dict(self, d: dict) -> None:
        self._mean = d["mean"]
        self._var = d["var"]
        self._count = d["count"]
        self._ret = d["ret"]

    def _update(self, val: float) -> None:
        self._count += 1
        if self._count == 1:
            self._mean = val
            self._var = 0.0
            return
        delta = val - self._mean
        self._mean += delta / self._count
        delta2 = val - self._mean
        self._var += (delta * delta2 - self._var) / self._count


class RunningObsNormalizer:
    """Per-dimension EMA normalizer for observation vectors.

    Maintains an exponential moving mean and variance for every feature
    dimension so the Transformer backbone sees ~N(0, 1) inputs throughout
    training, regardless of raw indicator scales.

    Designed to be serialized into the checkpoint alongside the model so
    inference can apply the same normalization as training.

    Args:
        obs_dim: Observation vector length.
        alpha:   EMA decay; 0.001 ≈ adapts over ~1 000 steps.
        clip:    Symmetric clamp after normalization (prevents gradient spikes).
        warmup:  Return raw obs for this many steps while stats accumulate.
    """

    def __init__(
        self,
        obs_dim: int,
        alpha: float = 0.001,
        clip: float = 5.0,
        warmup: int = 1000,
    ) -> None:
        self._alpha = alpha
        self._clip = clip
        self._warmup = warmup
        self._count = 0
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._var = np.ones(obs_dim, dtype=np.float64)

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize *obs* and optionally update running statistics.

        Pass ``update=False`` during evaluation / inference to freeze
        the normalizer at its training-time state.
        """
        if update:
            self._count += 1
            x = obs.astype(np.float64)
            delta = x - self._mean
            self._mean += self._alpha * delta
            self._var = (
                (1.0 - self._alpha) * self._var
                + self._alpha * delta * (x - self._mean)
            )
            self._var = np.maximum(self._var, 1e-6)

        if self._count <= self._warmup:
            return obs  # return raw during warm-up

        std = np.sqrt(self._var)
        normed = (obs.astype(np.float64) - self._mean) / (std + 1e-8)
        return np.clip(normed, -self._clip, self._clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "count": self._count,
            "mean": self._mean.tolist(),
            "var": self._var.tolist(),
            "alpha": self._alpha,
            "clip": self._clip,
            "warmup": self._warmup,
        }

    def load_state_dict(self, d: dict) -> None:
        self._count = int(d["count"])
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._var = np.array(d["var"], dtype=np.float64)
        self._alpha = float(d.get("alpha", self._alpha))
        self._clip = float(d.get("clip", self._clip))
        self._warmup = int(d.get("warmup", self._warmup))
