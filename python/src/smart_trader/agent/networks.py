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
    """Shared feature extractor using Transformer encoder layers.

    Supports two modes controlled by ``lookback``:

    * **lookback=1** (legacy): the full flat observation is projected as a
      single token.  Backward-compatible with v7 and earlier checkpoints.
    * **lookback>1** (sequence): the flat observation is split into a market
      feature sequence of ``lookback`` bars and a context vector (portfolio
      state + time embedding).  The context is projected as a learned
      ``[CTX]`` token prepended to the sequence, allowing cross-attention
      between current state and recent price history.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.05,
        lookback: int = 1,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.context_dim = context_dim
        self.output_dim = d_model

        if lookback > 1 and context_dim > 0:
            self.token_dim = (input_dim - context_dim) // lookback
            self.projection = nn.Linear(self.token_dim, d_model)
            self.context_proj = nn.Linear(context_dim, d_model)
        else:
            self.token_dim = input_dim
            self.projection = nn.Linear(input_dim, d_model)
            self.context_proj = None

        self.pos_enc = PositionalEncoding(d_model, max_len=max(lookback + 2, 512))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, flat_dim) — flat observation vector.
        Returns:
            (batch, d_model) — pooled representation.
        """
        if self.lookback > 1 and self.context_proj is not None:
            ctx = x[:, -self.context_dim:]
            market = x[:, :-self.context_dim]
            market = market.view(-1, self.lookback, self.token_dim)
            market = self.projection(market)
            ctx_token = self.context_proj(ctx).unsqueeze(1)
            seq = torch.cat([ctx_token, market], dim=1)
        else:
            if x.dim() == 2:
                seq = x.unsqueeze(1)
            else:
                seq = x
            seq = self.projection(seq)

        seq = self.pos_enc(seq)
        seq = self.encoder(seq)
        return seq.mean(dim=1)


class PolicyHead(nn.Module):
    """Hybrid policy head outputting continuous + discrete actions.

    Outputs:
      - position mu/sigma: (batch, 1) each — tanh-squashed Gaussian for [-1, 1]
      - risk mu/sigma:  (batch, 1) each — Gaussian for risk budget [0.01, 0.10]
      - hold logits:    (batch, 3) — categorical over {15m, 1h, 4h}
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_mu = nn.Linear(d_model, 1)
        self.position_log_std = nn.Parameter(torch.zeros(1))
        self.risk_mu = nn.Linear(d_model, 1)
        self.risk_log_std = nn.Parameter(torch.zeros(1))
        self.hold_head = nn.Linear(d_model, 3)

    def forward(self, features: torch.Tensor):
        features = self.shared(features)
        pos_mu = self.position_mu(features)
        pos_std = torch.exp(self.position_log_std).expand_as(pos_mu)
        risk_mu = torch.sigmoid(self.risk_mu(features)) * 0.09 + 0.01
        risk_std = torch.exp(self.risk_log_std).expand_as(risk_mu)
        hold_logits = self.hold_head(features)
        return pos_mu, pos_std, risk_mu, risk_std, hold_logits

    @staticmethod
    def _tanh_log_prob_correction(raw: torch.Tensor) -> torch.Tensor:
        """Jacobian correction for tanh squashing: -log(1 - tanh^2(x))."""
        return -torch.log(1.0 - torch.tanh(raw).pow(2) + 1e-6)

    def sample(
        self, features: torch.Tensor,
    ) -> Tuple[dict, torch.Tensor]:
        """Sample actions and compute log-probabilities."""
        pos_mu, pos_std, risk_mu, risk_std, hold_logits = self(features)

        pos_dist = Normal(pos_mu, pos_std)
        pos_raw = pos_dist.sample()
        position = torch.tanh(pos_raw)

        risk_dist = Normal(risk_mu, risk_std)
        risk_raw = risk_dist.sample()
        risk_budget = torch.clamp(risk_raw, 0.01, 0.10)

        hold_dist = Categorical(logits=hold_logits)
        hold = hold_dist.sample()

        log_prob = (
            pos_dist.log_prob(pos_raw).sum(-1)
            - self._tanh_log_prob_correction(pos_raw).sum(-1)
            + risk_dist.log_prob(risk_raw).sum(-1)
            + hold_dist.log_prob(hold)
        )

        action = {
            "position": position.cpu().detach().numpy(),
            "risk_budget": risk_budget.cpu().detach().numpy(),
            "hold_bars": hold.cpu().numpy(),
            "_pos_raw": pos_raw.cpu().detach().numpy(),
            "_risk_raw": risk_raw.cpu().detach().numpy(),
        }
        return action, log_prob

    def log_prob(
        self, features: torch.Tensor, actions: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob and entropy for given actions (for PPO update)."""
        pos_mu, pos_std, risk_mu, risk_std, hold_logits = self(features)

        pos_dist = Normal(pos_mu, pos_std)
        risk_dist = Normal(risk_mu, risk_std)
        hold_dist = Categorical(logits=hold_logits)

        pos_raw_t = torch.as_tensor(actions["_pos_raw"], dtype=torch.float32, device=features.device)
        risk_raw_t = torch.as_tensor(actions["_risk_raw"], dtype=torch.float32, device=features.device)
        hold_t = torch.as_tensor(actions["hold_bars"], dtype=torch.long, device=features.device)

        log_p = (
            pos_dist.log_prob(pos_raw_t).sum(-1)
            - self._tanh_log_prob_correction(pos_raw_t).sum(-1)
            + risk_dist.log_prob(risk_raw_t).sum(-1)
            + hold_dist.log_prob(hold_t)
        )
        entropy = (
            pos_dist.entropy().sum(-1)
            + risk_dist.entropy().sum(-1)
            + hold_dist.entropy()
        )
        return log_p, entropy


class ValueHead(nn.Module):
    """Value function head — predicts expected return."""

    def __init__(self, d_model: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
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
        lookback: int = 1,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(
            obs_dim, d_model, n_heads, n_layers,
            lookback=lookback, context_dim=context_dim,
        )
        self.policy = PolicyHead(d_model)
        self.value = ValueHead(d_model)

    def forward(self, obs: torch.Tensor) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        action, log_prob = self.policy.sample(features)
        value = self.value(features)
        return action, log_prob, value
