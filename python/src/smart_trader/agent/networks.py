"""Neural network architectures for the Meta Controller.

Implements a shared Transformer-encoder backbone with separate policy
and value heads.  The policy head outputs a hybrid discrete+continuous
action via a mixture of categorical and Gaussian distributions.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    """Shared feature extractor using Transformer encoder layers."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
        Returns:
            (batch, d_model) — pooled representation
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # treat as length-1 sequence
        x = self.projection(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x.mean(dim=1)  # global average pooling


class PolicyHead(nn.Module):
    """Hybrid policy head outputting discrete + continuous actions.

    Outputs:
      - regime logits:  (batch, 4) — categorical over {up, down, range, uncertain}
      - position mu/sigma: (batch, 1) each — Gaussian for target position [-1, 1]
      - risk mu/sigma:  (batch, 1) each — Gaussian for risk budget [0.01, 0.10]
      - hold logits:    (batch, 3) — categorical over {1h, 4h, 1d}
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.regime_head = nn.Linear(d_model, 4)
        self.position_mu = nn.Linear(d_model, 1)
        self.position_log_std = nn.Parameter(torch.zeros(1))
        self.risk_mu = nn.Linear(d_model, 1)
        self.risk_log_std = nn.Parameter(torch.zeros(1))
        self.hold_head = nn.Linear(d_model, 3)

    def forward(self, features: torch.Tensor):
        regime_logits = self.regime_head(features)
        pos_mu = torch.tanh(self.position_mu(features))
        pos_std = torch.exp(self.position_log_std).expand_as(pos_mu)
        risk_mu = torch.sigmoid(self.risk_mu(features)) * 0.09 + 0.01
        risk_std = torch.exp(self.risk_log_std).expand_as(risk_mu)
        hold_logits = self.hold_head(features)
        return regime_logits, pos_mu, pos_std, risk_mu, risk_std, hold_logits

    def sample(
        self, features: torch.Tensor,
    ) -> Tuple[dict, torch.Tensor]:
        """Sample actions and compute log-probabilities."""
        regime_logits, pos_mu, pos_std, risk_mu, risk_std, hold_logits = self(features)

        regime_dist = Categorical(logits=regime_logits)
        regime = regime_dist.sample()

        pos_dist = Normal(pos_mu, pos_std)
        pos_raw = pos_dist.sample()
        position = torch.tanh(pos_raw)

        risk_dist = Normal(risk_mu, risk_std)
        risk_raw = risk_dist.sample()
        risk_budget = torch.clamp(risk_raw, 0.01, 0.10)

        hold_dist = Categorical(logits=hold_logits)
        hold = hold_dist.sample()

        log_prob = (
            regime_dist.log_prob(regime)
            + pos_dist.log_prob(pos_raw).sum(-1)
            + risk_dist.log_prob(risk_raw).sum(-1)
            + hold_dist.log_prob(hold)
        )

        action = {
            "regime": regime.cpu().numpy(),
            "position": position.cpu().detach().numpy(),
            "risk_budget": risk_budget.cpu().detach().numpy(),
            "hold_bars": hold.cpu().numpy(),
        }
        return action, log_prob

    def log_prob(
        self, features: torch.Tensor, actions: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob and entropy for given actions (for PPO update)."""
        regime_logits, pos_mu, pos_std, risk_mu, risk_std, hold_logits = self(features)

        regime_dist = Categorical(logits=regime_logits)
        pos_dist = Normal(pos_mu, pos_std)
        risk_dist = Normal(risk_mu, risk_std)
        hold_dist = Categorical(logits=hold_logits)

        regime_t = torch.as_tensor(actions["regime"], dtype=torch.long, device=features.device)
        pos_t = torch.as_tensor(actions["position"], dtype=torch.float32, device=features.device)
        risk_t = torch.as_tensor(actions["risk_budget"], dtype=torch.float32, device=features.device)
        hold_t = torch.as_tensor(actions["hold_bars"], dtype=torch.long, device=features.device)

        log_p = (
            regime_dist.log_prob(regime_t)
            + pos_dist.log_prob(pos_t).sum(-1)
            + risk_dist.log_prob(risk_t).sum(-1)
            + hold_dist.log_prob(hold_t)
        )
        entropy = (
            regime_dist.entropy()
            + pos_dist.entropy().sum(-1)
            + risk_dist.entropy().sum(-1)
            + hold_dist.entropy()
        )
        return log_p, entropy


class ValueHead(nn.Module):
    """Value function head — predicts expected return."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class MetaControllerNetwork(nn.Module):
    """Complete Meta Controller: backbone + policy + value heads."""

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(obs_dim, d_model, n_heads, n_layers)
        self.policy = PolicyHead(d_model)
        self.value = ValueHead(d_model)

    def forward(self, obs: torch.Tensor) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        action, log_prob = self.policy.sample(features)
        value = self.value(features)
        return action, log_prob, value
