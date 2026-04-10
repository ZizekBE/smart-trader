"""Reward engine for RL training.

Implements the composite reward function:

    R_t = α·risk_adjusted_return + β·drawdown_penalty
        + γ·trading_cost_penalty  + δ·holding_bonus

All components are bounded to prevent extreme values that destabilise
PPO's value function.  An optional RunningNormalizer can wrap the
final scalar reward into ~N(0,1) for stable training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardConfig:
    alpha: float = 1.0        # risk-adjusted return weight
    beta: float = 0.5         # drawdown penalty weight
    gamma: float = 0.3        # trading cost penalty weight
    delta: float = 0.05       # holding bonus weight
    trade_penalty: float = 0.15  # flat penalty per trade (discourages churn)

    dd_threshold: float = 0.05       # drawdown % at which penalty kicks in
    dd_exponent: float = 2.0         # non-linear drawdown penalty exponent
    sortino_window: int = 20         # lookback for Sortino calculation
    target_return: float = 0.0       # Sortino minimum acceptable return

    clip_reward: float = 10.0        # symmetric per-step reward clip


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

        All sub-components are bounded so the raw reward stays in
        a reasonable range even under extreme market moves or leverage.
        """
        step_return = (portfolio_value - prev_value) / (prev_value + 1e-9)
        self.state.returns.append(step_return)

        # --- risk-adjusted return (incremental Sortino) ---
        risk_adj = self._sortino_increment(step_return)
        risk_adj = float(np.clip(risk_adj, -5.0, 5.0))

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
            cost_penalty = -trade_cost / (prev_value + 1e-9)
            cost_penalty = max(cost_penalty, -1.0)
            self.state.n_trades += 1

        # --- holding bonus ---
        holding_bonus = 0.0
        if position_held:
            self.state.holding_steps += 1
            holding_bonus = self.cfg.delta

        # flat penalty each time a trade fires — makes agent think twice
        churn_penalty = -self.cfg.trade_penalty if did_trade else 0.0

        reward = (
            self.cfg.alpha * risk_adj
            + self.cfg.beta * dd_penalty
            + self.cfg.gamma * cost_penalty
            + holding_bonus
            + churn_penalty
        )
        return float(np.clip(reward, -self.cfg.clip_reward, self.cfg.clip_reward))

    def _sortino_increment(self, step_return: float) -> float:
        """Incremental Sortino-style risk-adjusted return."""
        returns = self.state.returns
        if len(returns) < 2:
            return float(np.clip(step_return * 100, -5.0, 5.0))

        window = returns[-self.cfg.sortino_window:]
        downside = [min(r - self.cfg.target_return, 0) for r in window]
        downside_dev = float(np.sqrt(np.mean(np.square(downside)) + 1e-9))
        return step_return / (downside_dev + 1e-4)

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
