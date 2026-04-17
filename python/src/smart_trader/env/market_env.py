"""MarketEnv — Gymnasium-compatible trading environment for RL training.

Replays historical OHLCV data through the MarketSimulator, computes
features via the FeatureEngine, and uses the RewardEngine for composite
reward shaping.  Supports both spot and futures (with funding + liquidation).

Usage::

    env = MarketEnv(MarketEnvConfig(
        data={"1m": df_1m, "5m": df_5m, "1h": df_1h, "4h": df_4h},
    ))
    obs, info = env.reset()
    for _ in range(10_000):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from smart_trader.data.features.engine import FeatureConfig, compute_features
from smart_trader.env.reward import RewardConfig, RewardEngine, RunningObsNormalizer
from smart_trader.env.simulator import (
    FillModel,
    MarketSimulator,
    Position,
    SimulatorConfig,
)
from smart_trader.env.spaces import SpaceConfig, build_action_space, build_observation_space

_HOLD_BARS_MAP = {0: 15, 1: 60, 2: 240}  # 15m / 1h / 4h in 1m bars


@dataclass
class MarketEnvConfig:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    datasets: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    primary_tf: str = "1m"
    initial_cash: float = 10_000.0
    max_episode_bars: int = 24 * 60      # 1 day of 1m bars
    leverage_enabled: bool = True
    max_leverage: float = 3.0
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    sim_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    space_config: SpaceConfig = field(default_factory=SpaceConfig)
    seed: int = 42
    # Observation normalization (EMA per-dimension z-score)
    normalize_obs: bool = True
    obs_norm_alpha: float = 0.001   # EMA step; adapts over ~1 000 steps
    obs_norm_clip: float = 5.0      # symmetric clamp after normalization
    obs_norm_warmup: int = 1000     # steps before normalization activates
    # Microstructure features (optional; requires data coverage gate to pass)
    # funding_rates: {symbol → pd.Series(rate, index=time)} from funding_rates table
    # open_interest: {symbol → pd.Series(value_usd, index=time)} — deferred until ≥6 months
    funding_rates: dict | None = None  # dict[str, pd.Series]


class MarketEnv(gym.Env):
    """Gymnasium environment for hierarchical RL trading."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MarketEnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MarketEnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        self.observation_space = build_observation_space(self.cfg.space_config)
        self.action_space = build_action_space()

        self._sim = MarketSimulator(self.cfg.sim_config)
        self._reward_engine = RewardEngine(self.cfg.reward_config)

        self._build_datasets()
        # Funding rate lookup: {symbol → pd.Series sorted by time}
        self._fr_by_sym: dict[str, pd.Series] = {}
        if self.cfg.funding_rates:
            for sym, s in self.cfg.funding_rates.items():
                self._fr_by_sym[sym] = s.sort_index()
        self._cur_fr: pd.Series | None = None

        self._select_symbol(list(self._all_features.keys())[0])
        self._reset_state()

        obs_dim = self.observation_space.shape[0]
        self._obs_norm: RunningObsNormalizer | None = (
            RunningObsNormalizer(
                obs_dim,
                alpha=self.cfg.obs_norm_alpha,
                clip=self.cfg.obs_norm_clip,
                warmup=self.cfg.obs_norm_warmup,
            )
            if self.cfg.normalize_obs else None
        )

    # ── Gym interface ──────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if len(self._symbols) > 1:
            sym = self._rng.choice(self._symbols)
            self._select_symbol(sym)

        self._reset_state()
        obs = self._get_observation()
        return obs, self._info()

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev_value = self._portfolio_value()

        target_pos = float(np.asarray(action["position"]).flat[0])
        target_pos = float(np.clip(target_pos, -1, 1))
        risk_budget = float(np.asarray(action["risk_budget"]).flat[0])
        risk_budget = float(np.clip(risk_budget, 0.01, 0.10))
        hold_bars = _HOLD_BARS_MAP.get(int(np.asarray(action["hold_bars"]).flat[0]), 1)

        total_cost = 0.0
        did_trade = False

        # execute for hold_bars steps (or until episode ends)
        for _ in range(hold_bars):
            if self._step >= self._max_step:
                break

            bar = self._get_bar(self._step)
            cost, traded = self._execute_target(target_pos, risk_budget, bar)
            total_cost += cost
            did_trade = did_trade or traded

            self._update_position_pnl(bar["close"])
            self._step += 1

        curr_value = self._portfolio_value()

        reward = self._reward_engine.step(
            portfolio_value=curr_value,
            prev_value=prev_value,
            trade_cost=total_cost,
            did_trade=did_trade,
            position_held=self._position.is_open,
        )

        terminated = self._check_terminated()
        truncated = self._step >= self._max_step

        if terminated or truncated:
            reward += self._reward_engine.terminal_dd_reward()

        obs = self._get_observation()
        return obs, reward, terminated, truncated, self._info()

    # ── internal state ─────────────────────────────────────────

    def _reset_state(self) -> None:
        self._cash = self.cfg.initial_cash
        self._position = Position(symbol="", side="flat")

        data_len = len(self._primary_data) - 1
        latest_start = max(self._warmup_bars, data_len - self.cfg.max_episode_bars)
        if latest_start > self._warmup_bars:
            self._step = int(self._rng.integers(self._warmup_bars, latest_start + 1))
        else:
            self._step = self._warmup_bars

        self._max_step = min(data_len, self._step + self.cfg.max_episode_bars)
        self._trade_log: list[dict] = []
        self._reward_engine.reset(self.cfg.initial_cash)

    def _build_datasets(self) -> None:
        """Pre-compute features for all symbols and timeframes."""
        symbol_map: dict[str, dict[str, pd.DataFrame]] = {}
        if self.cfg.datasets:
            symbol_map = self.cfg.datasets
        elif self.cfg.data:
            symbol_map = {"_default": self.cfg.data}

        self._symbols: list[str] = []
        self._all_features: dict[str, dict[str, pd.DataFrame]] = {}
        self._all_primary: dict[str, pd.DataFrame] = {}

        for sym, tf_map in symbol_map.items():
            feats: dict[str, pd.DataFrame] = {}
            for tf, df in tf_map.items():
                if df.empty:
                    continue
                feats[tf] = compute_features(df, self.cfg.feature_config, prefix=f"{tf}_")
            primary = tf_map.get(self.cfg.primary_tf, pd.DataFrame())
            if primary.empty or not feats:
                continue
            self._symbols.append(sym)
            self._all_features[sym] = feats
            self._all_primary[sym] = primary

    def _select_symbol(self, sym: str) -> None:
        """Set the active symbol for the current episode."""
        self._active_sym = sym
        self._features = self._all_features[sym]
        self._primary_data = self._all_primary[sym]
        self._warmup_bars = min(200, len(self._primary_data) // 4)
        self._cur_fr = self._fr_by_sym.get(sym)

    def _get_bar(self, idx: int) -> dict:
        row = self._primary_data.iloc[idx]
        return {
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    # ── observation ────────────────────────────────────────────

    def _bar_features(self, step: int) -> np.ndarray:
        """Market features for a single bar across all timeframes."""
        sc = self.cfg.space_config
        parts: list[np.ndarray] = []
        for tf in sorted(self._features.keys()):
            feat_df = self._features[tf]
            if step < len(feat_df):
                row = feat_df.iloc[step].to_numpy(dtype=np.float32, na_value=0.0)
            else:
                row = np.zeros(len(feat_df.columns), dtype=np.float32)
            padded = np.zeros(sc.features_per_tf, dtype=np.float32)
            n = min(len(row), sc.features_per_tf)
            padded[:n] = row[:n]
            parts.append(padded)
        while len(parts) < sc.n_timeframes:
            parts.append(np.zeros(sc.features_per_tf, dtype=np.float32))
        return np.concatenate(parts)

    def _get_observation(self) -> np.ndarray:
        sc = self.cfg.space_config
        lookback = sc.lookback
        market_dim = sc.market_dim

        # --- market feature sequence (lookback bars) ---
        bar_rows: list[np.ndarray] = []
        start = max(0, self._step - lookback + 1)
        for t in range(start, self._step + 1):
            bar_rows.append(self._bar_features(t))
        while len(bar_rows) < lookback:
            bar_rows.insert(0, np.zeros(market_dim, dtype=np.float32))
        market = np.concatenate(bar_rows)

        # --- context vector (portfolio state + microstructure + time) ---
        total = self._portfolio_value()
        pos_frac = 0.0
        if self._position.is_open:
            pos_frac = self._position.notional / (total + 1e-9)
            if self._position.side == "short":
                pos_frac = -pos_frac
        dd = (self._reward_engine.state.peak_value - total) / (
            self._reward_engine.state.peak_value + 1e-9
        )
        portfolio = np.array([
            pos_frac,
            self._position.unrealized_pnl / (total + 1e-9),
            self._position.margin / (total + 1e-9),
            self._cash / (total + 1e-9),
            self._position.leverage / self.cfg.max_leverage,
            dd,
        ], dtype=np.float32)

        if sc.microstructure_dim > 0:
            micro_vals: list[float] = []
            if self._step < len(self._primary_data):
                ts = self._primary_data.index[self._step]
                # Funding rate: 8h-interval series; asof() returns last entry ≤ ts
                if self._cur_fr is not None and len(self._cur_fr) > 0:
                    val = self._cur_fr.asof(ts)
                    micro_vals.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    micro_vals.append(0.0)
            else:
                micro_vals.extend([0.0] * sc.microstructure_dim)
            # Pad or trim to exactly microstructure_dim
            while len(micro_vals) < sc.microstructure_dim:
                micro_vals.append(0.0)
            micro = np.array(micro_vals[:sc.microstructure_dim], dtype=np.float32)
        else:
            micro = np.empty(0, dtype=np.float32)

        if self._step < len(self._primary_data):
            ts = self._primary_data.index[self._step]
            if isinstance(ts, pd.Timestamp):
                hour = ts.hour
                dow = ts.dayofweek
            else:
                hour, dow = 0, 0
        else:
            hour, dow = 0, 0
        time_emb = np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ], dtype=np.float32)

        context = np.concatenate([portfolio, micro, time_emb])

        obs = np.concatenate([market, context])
        expected = self.observation_space.shape[0]
        if len(obs) < expected:
            obs = np.pad(obs, (0, expected - len(obs)))
        elif len(obs) > expected:
            obs = obs[:expected]

        if self._obs_norm is not None:
            obs = self._obs_norm.normalize(obs)
        return obs

    # ── execution logic ────────────────────────────────────────

    def _execute_target(
        self, target_pos: float, risk_budget: float, bar: dict,
    ) -> tuple[float, bool]:
        """Move from current position toward target_pos fraction."""
        total = self._portfolio_value()
        target_notional = target_pos * total * min(
            self.cfg.max_leverage if self.cfg.leverage_enabled else 1.0,
            self.cfg.max_leverage,
        )

        current_notional = 0.0
        if self._position.is_open:
            sign = 1.0 if self._position.side == "long" else -1.0
            current_notional = sign * self._position.size * bar["close"]

        diff = target_notional - current_notional
        scale = max(abs(current_notional), abs(target_notional), total)
        if abs(diff) < scale * 0.10:
            return 0.0, False

        price = bar["close"]
        qty = abs(diff) / (price + 1e-9)
        side = "buy" if diff > 0 else "sell"

        max_risk_notional = total * risk_budget
        qty = min(qty, max_risk_notional / (price + 1e-9))

        fill = self._sim.simulate_fill(side, qty, bar)
        if fill.status == "rejected":
            return 0.0, False

        cost = fill.fee + fill.slippage_cost
        self._apply_fill(side, fill.fill_qty, fill.fill_price, cost)
        self._trade_log.append({
            "step": self._step, "side": side, "qty": fill.fill_qty,
            "price": fill.fill_price, "cost": cost,
        })
        return cost, True

    def _apply_fill(self, side: str, qty: float, price: float, cost: float) -> None:
        """Update position and cash after a fill."""
        if not self._position.is_open:
            notional = qty * price
            max_lev = self.cfg.max_leverage if self.cfg.leverage_enabled else 1.0
            leverage = min(notional / (self._cash + 1e-9), max_lev)
            leverage = max(leverage, 1.0)

            self._position.side = "long" if side == "buy" else "short"
            self._position.size = qty
            self._position.entry_price = price
            self._position.leverage = leverage
            self._cash -= cost
            if leverage <= 1.0:
                self._cash -= notional
            else:
                margin = notional / leverage
                self._position.margin = margin
                self._cash -= margin
            return

        same_side = (
            (side == "buy" and self._position.side == "long")
            or (side == "sell" and self._position.side == "short")
        )
        if same_side:
            old_notional = self._position.size * self._position.entry_price
            new_notional = qty * price
            self._position.entry_price = (
                (old_notional + new_notional) / (self._position.size + qty + 1e-12)
            )
            self._position.size += qty
            self._cash -= qty * price + cost
        else:
            close_qty = min(qty, self._position.size)
            pnl_per_unit = price - self._position.entry_price
            if self._position.side == "short":
                pnl_per_unit = -pnl_per_unit
            realized = close_qty * pnl_per_unit
            self._cash += close_qty * self._position.entry_price + realized - cost
            self._position.size -= close_qty

            if self._position.size < 1e-12:
                remaining = qty - close_qty
                if remaining > 1e-12:
                    self._position.side = "long" if side == "buy" else "short"
                    self._position.size = remaining
                    self._position.entry_price = price
                    self._cash -= remaining * price
                else:
                    self._position = Position(symbol="", side="flat")

    def _update_position_pnl(self, price: float) -> None:
        if not self._position.is_open:
            return
        diff = price - self._position.entry_price
        if self._position.side == "short":
            diff = -diff
        self._position.unrealized_pnl = diff * self._position.size

    def _portfolio_value(self) -> float:
        return self._cash + (
            self._position.unrealized_pnl + self._position.size * self._position.entry_price
            if self._position.is_open else 0.0
        )

    def _check_terminated(self) -> bool:
        val = self._portfolio_value()
        if val <= self.cfg.initial_cash * 0.70:
            return True
        if self._position.is_open:
            bar = self._get_bar(min(self._step, len(self._primary_data) - 1))
            if self._sim.check_liquidation(self._position, bar["close"]):
                return True
        return False

    def _info(self) -> dict:
        return {
            "step": self._step,
            "symbol": getattr(self, "_active_sym", ""),
            "cash": self._cash,
            "portfolio_value": self._portfolio_value(),
            "position_side": self._position.side,
            "position_size": self._position.size,
            "n_trades": len(self._trade_log),
            **self._reward_engine.get_episode_stats(),
        }
