"""ObservationBuilder — construct MarketEnv-compatible observations for inference.

Bridges the gap between FeatureStore output and the full observation vector
expected by the trained Meta Controller (market feature sequence + context).
Supports both legacy flat (lookback=1) and sequence (lookback>1) layouts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smart_trader.env.spaces import SpaceConfig


@dataclass
class PortfolioSnapshot:
    position_fraction: float = 0.0
    unrealized_pnl_frac: float = 0.0
    margin_frac: float = 0.0
    cash_frac: float = 1.0
    leverage_frac: float = 0.0
    drawdown: float = 0.0


class ObservationBuilder:
    """Builds a flat observation vector matching MarketEnv._get_observation.

    Maintains a sliding window of past bar features for sequence mode.
    """

    def __init__(self, space_cfg: SpaceConfig | None = None) -> None:
        self.cfg = space_cfg or SpaceConfig()
        self.obs_dim = self.cfg.lookback * self.cfg.market_dim + self.cfg.context_dim
        self._history: deque[np.ndarray] = deque(maxlen=self.cfg.lookback)

    def reset(self) -> None:
        """Clear the lookback history (call on episode/session reset)."""
        self._history.clear()

    def _bar_vector(self, tf_features: dict[str, np.ndarray]) -> np.ndarray:
        """Build a single bar's market features across all TFs."""
        parts: list[np.ndarray] = []
        for tf in sorted(tf_features.keys()):
            row = tf_features[tf]
            padded = np.zeros(self.cfg.features_per_tf, dtype=np.float32)
            n = min(len(row), self.cfg.features_per_tf)
            padded[:n] = row[:n]
            parts.append(padded)
        while len(parts) < self.cfg.n_timeframes:
            parts.append(np.zeros(self.cfg.features_per_tf, dtype=np.float32))
        return np.concatenate(parts)

    @staticmethod
    def _time_embedding(timestamp: pd.Timestamp | None) -> np.ndarray:
        if timestamp is not None and isinstance(timestamp, pd.Timestamp):
            hour, dow = timestamp.hour, timestamp.dayofweek
        else:
            hour, dow = 0, 0
        return np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ], dtype=np.float32)

    def build(
        self,
        tf_features: dict[str, np.ndarray],
        portfolio: PortfolioSnapshot | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> np.ndarray:
        """Construct the full observation vector.

        Args:
            tf_features: {timeframe: 1-D feature array} from FeatureEngine.
            portfolio:   Current portfolio state.  Defaults to neutral.
            timestamp:   Current bar timestamp for time embedding.
        """
        portfolio = portfolio or PortfolioSnapshot()
        bar = self._bar_vector(tf_features)
        self._history.append(bar)

        # --- market sequence ---
        lookback = self.cfg.lookback
        market_dim = self.cfg.market_dim
        bars = list(self._history)
        while len(bars) < lookback:
            bars.insert(0, np.zeros(market_dim, dtype=np.float32))
        market = np.concatenate(bars)

        # --- context ---
        port_vec = np.array([
            portfolio.position_fraction,
            portfolio.unrealized_pnl_frac,
            portfolio.margin_frac,
            portfolio.cash_frac,
            portfolio.leverage_frac,
            portfolio.drawdown,
        ], dtype=np.float32)

        micro = np.zeros(self.cfg.microstructure_dim, dtype=np.float32) if self.cfg.microstructure_dim > 0 else np.empty(0, dtype=np.float32)
        time_emb = self._time_embedding(timestamp)
        context = np.concatenate([port_vec, micro, time_emb])

        obs = np.concatenate([market, context])
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
        return obs
