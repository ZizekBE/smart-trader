"""RL Trading Loop — production loop driven by the Meta Controller.

Replaces the rule-based TradingLoop with an RL-driven decision engine.
The Meta Controller outputs high-level decisions (regime, position target,
risk budget), while the deterministic execution layer handles order
placement, position management, and hard risk limits.

Supports both live and shadow mode.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from smart_trader.agent.inference import InferenceConfig, InferenceService, ShadowMode
from smart_trader.agent.observation import ObservationBuilder, PortfolioSnapshot
from smart_trader.core.settings import get_settings
from smart_trader.data.features.store import FeatureStore
from smart_trader.env.spaces import SpaceConfig
from smart_trader.exchange.base import ExchangeAdapter
from smart_trader.exchange.feed import FeedManager
from smart_trader.execution.position_manager import PositionManager
from smart_trader.execution.risk_guard import RiskGuard, RiskLimits

_REGIME_UPDATE_BARS = 24  # recompute regime every N 1h bars

log = structlog.get_logger(__name__)



class RLTradingLoop:
    """Production trading loop powered by the RL Meta Controller."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        feature_store: FeatureStore,
        inference_config: InferenceConfig,
        risk_limits: RiskLimits | None = None,
        symbols: list[str] | None = None,
        timeframes: list[str] = ("1m", "5m", "1h", "4h"),
        initial_capital: float = 10_000.0,
        space_config: SpaceConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._feature_store = feature_store
        self._inference = InferenceService(inference_config)
        self._pos_mgr = PositionManager(initial_capital=initial_capital)
        self._risk_guard = RiskGuard(limits=risk_limits, initial_capital=initial_capital)
        self._feed = FeedManager(adapter)
        self._symbols = symbols or get_settings().symbols
        self._timeframes = list(timeframes)
        self._space_cfg = space_config or SpaceConfig()
        self._obs_builder = ObservationBuilder(self._space_cfg)
        self._shadow: Optional[ShadowMode] = None
        self._running = False
        self._log = log.bind(loop="rl")

        # regime context cache: {symbol → (regime_vec, bar_count)}
        self._regime_cache: dict[str, tuple[np.ndarray, int]] = {}

        if inference_config.shadow_mode:
            self._shadow = ShadowMode(self._inference)

    async def start(self) -> None:
        """Initialize and run the trading loop."""
        self._log.info("rl_loop_starting", symbols=self._symbols)
        await self._inference.start()

        for symbol in self._symbols:
            self._feed.on_candle(symbol, self._timeframes[0], self._on_candle)

        await self._feed.start()
        self._running = True

        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        await self._feed.stop()
        await self._adapter.close()
        self._log.info("rl_loop_stopped")

    async def _get_regime_vec(self, symbol: str, ts: pd.Timestamp) -> np.ndarray | None:
        """Return cached regime vector, recomputing every _REGIME_UPDATE_BARS bars."""
        if self._space_cfg.regime_dim == 0:
            return None

        cached_vec, bar_count = self._regime_cache.get(symbol, (None, 0))
        bar_count += 1

        if cached_vec is not None and bar_count % _REGIME_UPDATE_BARS != 0:
            self._regime_cache[symbol] = (cached_vec, bar_count)
            return cached_vec

        try:
            from smart_trader.env.regime_features import compute_regime_features

            hourly_df = await self._feature_store.get_candles(symbol, "1h", lookback=2000)
            daily_df = await self._feature_store.get_candles(symbol, "1d", lookback=120)
            if hourly_df is None or hourly_df.empty:
                raise ValueError("no hourly data")

            primary_row = pd.DataFrame({"close": [1.0]}, index=[ts])
            reg_df = compute_regime_features(
                daily_df if daily_df is not None else pd.DataFrame(),
                hourly_df, primary_row, symbol, stride=6,
            )
            row = reg_df.iloc[0]
            vec = np.array([row["regime_code"], row["vol_code"]], dtype=np.float32)
            self._regime_cache[symbol] = (vec, bar_count)
            self._log.debug("regime_updated", symbol=symbol,
                            regime_code=float(vec[0]), vol_code=float(vec[1]))
            return vec
        except Exception as exc:
            self._log.warning("regime_compute_failed", symbol=symbol, error=str(exc))
            if cached_vec is not None:
                self._regime_cache[symbol] = (cached_vec, bar_count)
                return cached_vec
            return np.full(self._space_cfg.regime_dim, 0.5, dtype=np.float32)

    async def _build_observation(self, symbol: str, candle) -> np.ndarray:
        """Build a full observation vector matching training format."""
        tf_features: dict[str, np.ndarray] = {}
        for tf in self._timeframes:
            feat_df = await self._feature_store.get_features(symbol, tf, lookback=1)
            if not feat_df.empty:
                tf_features[tf] = feat_df.iloc[-1].to_numpy(dtype=np.float32, na_value=0.0)

        pos_info = self._pos_mgr.get_position(symbol)
        total_value = self._pos_mgr._capital
        portfolio = PortfolioSnapshot()
        if pos_info:
            portfolio = PortfolioSnapshot(
                position_fraction=pos_info.get("notional", 0) / (total_value + 1e-9),
                unrealized_pnl_frac=pos_info.get("unrealized_pnl", 0) / (total_value + 1e-9),
                margin_frac=pos_info.get("margin", 0) / (total_value + 1e-9),
                cash_frac=(total_value - pos_info.get("notional", 0)) / (total_value + 1e-9),
            )

        ts = pd.Timestamp(candle.time) if hasattr(candle, "time") else None
        regime_vec = await self._get_regime_vec(symbol, ts) if ts is not None else None
        return self._obs_builder.build(tf_features, portfolio, ts, regime_vec=regime_vec)

    async def _on_candle(self, candle) -> None:
        """Triggered on each new candle from the WebSocket feed."""
        symbol = candle.symbol
        try:
            obs = await self._build_observation(symbol, candle)

            if self._shadow:
                self._shadow.evaluate(obs)
                return

            result = self._inference.predict(obs)
            action = result.action

            target_pos = float(np.atleast_1d(action.get("position", 0))[0])
            risk_budget = float(np.atleast_1d(action.get("risk_budget", 0.02))[0])

            self._log.info(
                "agent_decision",
                symbol=symbol,
                target_pos=f"{target_pos:+.3f}",
                risk_budget=f"{risk_budget:.3f}",
                value=f"{result.value_estimate:.4f}",
                latency_ms=f"{result.latency_ms:.1f}",
            )

            ticker = await self._adapter.fetch_ticker(symbol)
            price = ticker.last

            delta = self._pos_mgr.compute_delta(
                symbol=symbol,
                target_fraction=target_pos,
                current_price=price,
                risk_budget=risk_budget,
            )

            if delta is None:
                return

            portfolio_value = self._pos_mgr._capital
            check = self._risk_guard.check(delta, price, portfolio_value)

            if not check.allowed:
                self._log.warning("order_rejected", reason=check.reason, symbol=symbol)
                return

            final_delta = check.adjusted_delta or delta

            order_result = await self._adapter.place_order(
                symbol=symbol,
                side=final_delta.side,
                amount=final_delta.quantity,
                order_type=final_delta.order_type,
            )

            if order_result.status == "filled":
                self._pos_mgr.update_position(
                    symbol, final_delta.side,
                    order_result.quantity, order_result.fill_price,
                )
                self._log.info(
                    "order_filled",
                    symbol=symbol, side=final_delta.side,
                    qty=order_result.quantity,
                    price=order_result.fill_price,
                    fee=order_result.fee,
                )
            else:
                self._log.warning(
                    "order_failed",
                    symbol=symbol, status=order_result.status,
                )

        except Exception as exc:
            self._log.error("tick_error", symbol=symbol, error=str(exc))
