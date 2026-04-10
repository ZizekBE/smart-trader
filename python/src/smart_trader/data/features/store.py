"""Feature Store — Redis-backed hot cache + DB persistence.

Stores pre-computed features in Redis for sub-ms access during live trading
and RL inference.  Falls back to on-the-fly computation from the candle DB
when cache misses occur.

Usage::

    store = FeatureStore(redis_url, db_session_factory)
    obs = await store.get_observation("BTC/USDT", timeframes=["1m", "1h", "4h"])
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import structlog

from smart_trader.data.features.engine import FeatureConfig, compute_features

log = structlog.get_logger(__name__)

_DEFAULT_LOOKBACK = 300
_FEATURE_TTL_S = 300


class FeatureStore:
    """Redis-backed feature cache with DB fallback."""

    def __init__(
        self,
        redis_client=None,
        session_factory=None,
        config: FeatureConfig | None = None,
    ) -> None:
        self._redis = redis_client
        self._session_factory = session_factory
        self._config = config or FeatureConfig()
        self._log = log

    # ── public API ─────────────────────────────────────────────

    async def get_features(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = _DEFAULT_LOOKBACK,
    ) -> pd.DataFrame:
        """Get feature DataFrame for a single symbol/timeframe."""
        cache_key = f"feat:{symbol}:{timeframe}"

        if self._redis:
            cached = await self._try_cache(cache_key)
            if cached is not None and len(cached) >= lookback:
                return cached.tail(lookback)

        df = await self._load_candles(symbol, timeframe, lookback + 50)
        if df.empty:
            return pd.DataFrame()

        features = compute_features(df, self._config, prefix=f"{timeframe}_")

        if self._redis:
            await self._write_cache(cache_key, features)

        return features.tail(lookback)

    async def get_observation(
        self,
        symbol: str,
        timeframes: Sequence[str] = ("1m", "5m", "1h", "4h"),
        lookback: int = 1,
    ) -> np.ndarray:
        """Build a flat observation vector for RL from multi-TF features.

        Returns the latest `lookback` rows stacked into a 1-D array.
        """
        vectors = []
        for tf in timeframes:
            feat = await self.get_features(symbol, tf, lookback)
            if feat.empty:
                continue
            arr = feat.tail(lookback).to_numpy(dtype=np.float32, na_value=0.0)
            vectors.append(arr.flatten())

        if not vectors:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(vectors)

    async def update_from_candle(
        self,
        symbol: str,
        timeframe: str,
        candle_dict: dict,
    ) -> None:
        """Incrementally update the cache when a new candle arrives via WS."""
        cache_key = f"feat:{symbol}:{timeframe}"
        cached = await self._try_cache(cache_key) if self._redis else None
        if cached is None:
            return

        new_row = pd.DataFrame([candle_dict]).set_index("time")
        combined = pd.concat([cached[["open", "high", "low", "close", "volume"]], new_row])
        combined = combined[~combined.index.duplicated(keep="last")]
        features = compute_features(combined, self._config, prefix=f"{timeframe}_")

        if self._redis:
            await self._write_cache(cache_key, features)

    # ── internal ───────────────────────────────────────────────

    async def _load_candles(
        self, symbol: str, timeframe: str, limit: int,
    ) -> pd.DataFrame:
        """Load OHLCV from DB via the candle repository."""
        if self._session_factory is None:
            return pd.DataFrame()

        from smart_trader.data.models.candle import Candle
        from sqlalchemy import select

        async with self._session_factory() as session:
            q = (
                select(Candle)
                .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
                .order_by(Candle.time.desc())
                .limit(limit)
            )
            result = await session.execute(q)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        records = [
            {
                "time": r.time,
                "open": float(r.open), "high": float(r.high),
                "low": float(r.low), "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in reversed(rows)
        ]
        df = pd.DataFrame(records).set_index("time")
        return df

    async def _try_cache(self, key: str) -> Optional[pd.DataFrame]:
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            df = pd.DataFrame(data["rows"], columns=data["cols"])
            df.index = pd.to_datetime(data["idx"])
            return df
        except Exception as exc:
            self._log.debug("cache_miss", key=key, error=str(exc))
            return None

    async def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            payload = json.dumps({
                "cols": list(df.columns),
                "idx": [str(t) for t in df.index],
                "rows": df.values.tolist(),
            })
            await self._redis.set(key, payload, ex=_FEATURE_TTL_S)
        except Exception as exc:
            self._log.warning("cache_write_error", key=key, error=str(exc))
