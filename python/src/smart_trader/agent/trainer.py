"""PPO trainer for the Meta Controller.

Implements Proximal Policy Optimization with GAE, entropy bonus,
value clipping, and gradient norm clipping.

Usage::

    trainer = PPOTrainer(env, agent, PPOConfig())
    stats = trainer.train(total_timesteps=100_000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

from smart_trader.agent.meta_controller import MetaController
from smart_trader.env.market_env import MarketEnv
from smart_trader.env.reward import RunningNormalizer

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

log = structlog.get_logger(__name__)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    lr_end: float | None = None   # linearly decay to this; defaults to lr/10
    weight_decay: float = 1e-5    # L2 regularisation via AdamW
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.02
    entropy_coef_end: float = 0.005  # linearly decay entropy bonus
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048           # steps per rollout
    n_epochs: int = 10            # PPO epochs per update
    batch_size: int = 64
    target_kl: float | None = 0.03  # early stopping on KL divergence
    eval_freq: int = 10           # evaluate every N updates
    eval_episodes: int = 5        # episodes per evaluation
    patience: int = 10            # stop if no eval improvement for N evals


@dataclass
class RolloutBuffer:
    """Stores experience from one rollout for PPO update."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self) -> None:
        for lst in (self.observations, self.actions, self.log_probs,
                    self.rewards, self.values, self.dones):
            lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """Proximal Policy Optimization trainer for the Meta Controller."""

    def __init__(
        self,
        env: MarketEnv,
        agent: MetaController,
        config: PPOConfig | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.cfg = config or PPOConfig()
        self.optimizer = optim.AdamW(
            agent.network.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.buffer = RolloutBuffer()
        self._reward_norm = RunningNormalizer(gamma=self.cfg.gamma)
        self._entropy_coef = self.cfg.entropy_coef
        self._lr_end = self.cfg.lr_end if self.cfg.lr_end is not None else self.cfg.lr / 10

    def train(
        self,
        total_timesteps: int = 100_000,
        checkpoint_dir: str | Path | None = None,
        checkpoint_freq: int = 10,
        resume_step: int = 0,
    ) -> dict:
        """Run PPO training loop with early stopping.

        Args:
            resume_step: Starting step count (for resumed training).

        Returns:
            Summary statistics dict.
        """
        tb_writer = self._make_tb_writer(checkpoint_dir)

        self.agent.network.train()
        step = resume_step
        update = resume_step // self.cfg.n_steps if resume_step else 0
        episode_rewards: list[float] = []
        all_stats: list[dict] = []

        best_eval_reward = -float("inf")
        evals_without_improvement = 0
        stopped_early = False

        total_updates_expected = max(
            (total_timesteps - resume_step) // self.cfg.n_steps, 1,
        )

        while step < total_timesteps:
            rollout_reward = self._collect_rollout()
            episode_rewards.append(rollout_reward)
            step += len(self.buffer)

            progress = min(len(episode_rewards) / total_updates_expected, 1.0)
            self._entropy_coef = (
                self.cfg.entropy_coef
                + (self.cfg.entropy_coef_end - self.cfg.entropy_coef) * progress
            )

            new_lr = self.cfg.lr + (self._lr_end - self.cfg.lr) * progress
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            stats = self._update()
            stats["rollout_reward"] = rollout_reward
            stats["total_steps"] = step
            all_stats.append(stats)
            update += 1

            log.info(
                "ppo_update",
                update=update,
                steps=step,
                reward=f"{rollout_reward:.4f}",
                policy_loss=f"{stats['policy_loss']:.4f}",
                value_loss=f"{stats['value_loss']:.4f}",
                entropy=f"{stats['entropy']:.4f}",
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/rollout_reward", rollout_reward, step)
                tb_writer.add_scalar("train/policy_loss", stats["policy_loss"], step)
                tb_writer.add_scalar("train/value_loss", stats["value_loss"], step)
                tb_writer.add_scalar("train/entropy", stats["entropy"], step)
                tb_writer.add_scalar("train/lr", new_lr, step)
                tb_writer.add_scalar("train/entropy_coef", self._entropy_coef, step)
                if "approx_kl" in stats:
                    tb_writer.add_scalar("train/approx_kl", stats["approx_kl"], step)

            if checkpoint_dir and update % checkpoint_freq == 0:
                self.agent.save(
                    Path(checkpoint_dir) / f"checkpoint_{update}.pt",
                    optimizer=self.optimizer,
                    step=step,
                    extra={"reward_norm": self._reward_norm.state_dict()},
                )

            if update % self.cfg.eval_freq == 0:
                eval_reward = self._evaluate(self.cfg.eval_episodes)
                log.info("eval", update=update, steps=step,
                         eval_reward=f"{eval_reward:.4f}",
                         best=f"{best_eval_reward:.4f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/reward", eval_reward, step)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    evals_without_improvement = 0
                    if checkpoint_dir:
                        self.agent.save(
                            Path(checkpoint_dir) / "best_agent.pt",
                            optimizer=self.optimizer,
                            step=step,
                            extra={"reward_norm": self._reward_norm.state_dict()},
                        )
                        log.info("new_best_saved", eval_reward=f"{eval_reward:.4f}",
                                 step=step)
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= self.cfg.patience:
                        log.info("early_stopping",
                                 patience=self.cfg.patience,
                                 best_eval=f"{best_eval_reward:.4f}",
                                 step=step)
                        stopped_early = True
                        break

        if tb_writer is not None:
            tb_writer.close()

        return {
            "total_updates": update - (resume_step // self.cfg.n_steps if resume_step else 0),
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "stopped_early": stopped_early,
            "best_eval_reward": best_eval_reward,
            "updates": all_stats,
        }

    @staticmethod
    def _make_tb_writer(checkpoint_dir: str | Path | None) -> SummaryWriter | None:
        if checkpoint_dir is None:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(checkpoint_dir) / "tb_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            return SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            log.warning("tensorboard_not_available")
            return None

    def _evaluate(self, n_episodes: int = 5) -> float:
        """Run deterministic eval episodes and return mean reward."""
        self.agent.network.eval()
        total_rewards: list[float] = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action, _, _ = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_rewards.append(ep_reward)
        self.agent.network.train()
        return float(np.mean(total_rewards))

    def _collect_rollout(self) -> float:
        """Collect n_steps of experience into the buffer."""
        self.buffer.clear()
        obs, _ = self.env.reset()
        raw_reward_sum = 0.0

        for _ in range(self.cfg.n_steps):
            action, log_prob, value = self.agent.act(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            norm_reward = self._reward_norm.normalize(reward, done)

            self.buffer.observations.append(obs)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.rewards.append(norm_reward)
            self.buffer.values.append(value)
            self.buffer.dones.append(done)

            raw_reward_sum += reward
            obs = next_obs

            if done:
                obs, _ = self.env.reset()

        return raw_reward_sum

    def _update(self) -> dict:
        """Run PPO update epochs on the collected rollout."""
        advantages, returns = self._compute_gae()

        obs_arr = np.array(self.buffer.observations)
        old_log_probs = np.array(self.buffer.log_probs)
        old_values_arr = np.array(self.buffer.values)

        actions_batch = {
            k: np.array([a[k] for a in self.buffer.actions])
            for k in self.buffer.actions[0]
        }

        device = self.agent.device
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        old_lp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        old_val_t = torch.as_tensor(old_values_arr, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_arr)
        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for epoch in range(self.cfg.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                idx = indices[start:end]

                batch_obs = obs_t[idx]
                batch_adv = adv_t[idx]
                batch_ret = ret_t[idx]
                batch_old_lp = old_lp_t[idx]
                batch_old_val = old_val_t[idx]

                batch_actions = {k: v[idx] for k, v in actions_batch.items()}

                features = self.agent.network.backbone(batch_obs)
                new_lp, entropy = self.agent.network.policy.log_prob(features, batch_actions)
                new_values = self.agent.network.value(features)

                ratio = torch.exp(new_lp - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = batch_old_val + torch.clamp(
                    new_values - batch_old_val,
                    -self.cfg.value_clip, self.cfg.value_clip,
                )
                vl_unclipped = (new_values - batch_ret).pow(2)
                vl_clipped = (v_clipped - batch_ret).pow(2)
                value_loss = torch.max(vl_unclipped, vl_clipped).mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self._entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.network.parameters(), self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

            if self.cfg.target_kl is not None:
                with torch.no_grad():
                    features = self.agent.network.backbone(obs_t)
                    new_lp, _ = self.agent.network.policy.log_prob(features, actions_batch)
                    approx_kl = (old_lp_t - new_lp).mean().item()
                if approx_kl > self.cfg.target_kl:
                    break

        result: dict = {
            "policy_loss": total_pl / max(n_updates, 1),
            "value_loss": total_vl / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }
        if self.cfg.target_kl is not None:
            result["approx_kl"] = approx_kl  # type: ignore[possibly-undefined]
        return result

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else 0.0
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns
