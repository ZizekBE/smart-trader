"""Meta Controller — high-level RL agent for trading decisions.

Wraps MetaControllerNetwork with action selection, inference, and
model serialization.  Used both during training (with exploration)
and production (deterministic, exported to ONNX/TorchScript).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smart_trader.agent.networks import MetaControllerNetwork


class MetaController:
    """High-level hierarchical RL agent.

    Decides regime belief, target position, risk budget, and hold duration.
    Low-level execution is handled deterministically by the execution layer.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        device: str = "cpu",
        lookback: int = 1,
        context_dim: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.network = MetaControllerNetwork(
            obs_dim, d_model, n_heads, n_layers,
            lookback=lookback, context_dim=context_dim,
        )
        self.network.to(self.device)
        self._deterministic = False

    # ── action selection ───────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False,
    ) -> tuple[dict, float, float]:
        """Select action from observation.

        Returns:
            (action_dict, log_prob, value_estimate)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        features = self.network.backbone(obs_t)

        if deterministic or self._deterministic:
            action = self._deterministic_action(features)
            log_prob = 0.0
        else:
            action, lp = self.network.policy.sample(features)
            log_prob = float(lp.item())

        value = float(self.network.value(features).item())

        # squeeze away batch dim for single-obs inference
        action_out: dict = {}
        for k, v in action.items():
            if isinstance(v, np.ndarray):
                if k == "hold_bars":
                    action_out[k] = int(v.flat[0])
                else:
                    action_out[k] = v.flatten().astype(np.float32)
            else:
                action_out[k] = v

        # keep pre-squash values for PPO log-prob consistency
        if "_pos_raw" not in action_out:
            action_out["_pos_raw"] = np.zeros(1, dtype=np.float32)
        if "_risk_raw" not in action_out:
            action_out["_risk_raw"] = np.zeros(1, dtype=np.float32)
        return action_out, log_prob, value

    def set_deterministic(self, flag: bool = True) -> None:
        self._deterministic = flag

    def _deterministic_action(self, features: torch.Tensor) -> dict:
        pos_mu, _, risk_alpha, risk_beta, hold_logits = self.network.policy(features)
        # Beta mode = (α − 1) / (α + β − 2); valid since softplus+1 guarantees α, β > 1
        risk_mode_01 = (risk_alpha - 1.0) / (risk_alpha + risk_beta - 2.0 + 1e-8)
        risk_budget = risk_mode_01 * 0.09 + 0.01
        return {
            "position":    np.clip(np.tanh(pos_mu.cpu().numpy().flatten()), -1, 1),
            "risk_budget": np.clip(risk_budget.cpu().numpy().flatten(), 0.01, 0.10),
            "hold_bars":   int(hold_logits.argmax(-1).item()),
        }

    # ── serialization ──────────────────────────────────────────

    def save(self, path: str | Path, optimizer=None, step: int = 0,
             extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bb = self.network.backbone
        enc = bb.encoder
        n_layers = len(enc.layers)
        n_heads = int(enc.layers[0].self_attn.num_heads)
        obs_dim = bb.lookback * bb.token_dim + bb.context_dim if bb.lookback > 1 and bb.context_dim > 0 else bb.projection.in_features
        payload = {
            "model_state": self.network.state_dict(),
            "config": {
                "obs_dim": obs_dim,
                "d_model": bb.output_dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "lookback": bb.lookback,
                "context_dim": bb.context_dim,
            },
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str | Path, optimizer=None) -> dict:
        """Load checkpoint. Returns metadata dict (step, etc.)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return {k: v for k, v in ckpt.items()
                if k not in ("model_state", "optimizer_state")}

    def export_onnx(self, path: str | Path, obs_dim: int) -> None:
        """Export backbone+policy for low-latency inference."""
        self.network.eval()
        dummy = torch.randn(1, obs_dim, device=self.device)
        torch.onnx.export(
            self.network, dummy, str(path),
            input_names=["observation"],
            output_names=["position", "risk_budget", "hold_bars", "log_prob", "value"],
            dynamic_axes={"observation": {0: "batch"}},
        )

    def export_torchscript(self, path: str | Path, obs_dim: int) -> None:
        """Export as TorchScript for production serving."""
        self.network.eval()
        dummy = torch.randn(1, obs_dim, device=self.device)
        scripted = torch.jit.trace(self.network, dummy)
        scripted.save(str(path))
