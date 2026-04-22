"""TacticalSleeve — short-term momentum/mean-reversion positions.

Signal timeframe : 1h   (configurable)
Trend timeframe  : 1d
Mid timeframe    : 4h   (MTF alignment filter)
Confidence floor : 0.65
Stop style       : tighter ATR-based (from existing PositionSizer)
Take-profit      : fixed (existing RR from PositionSizer)
Time stop        : force-close after hybrid_short_time_stop_h hours
Max positions    : 3 per symbol-side pair
"""
from __future__ import annotations

import structlog
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from smart_trader.core.settings import get_settings
from smart_trader.risk.manager import RiskManager
from smart_trader.risk.models import Portfolio
from smart_trader.strategy.adaptive_params import RegimeParamAdapter
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer
from smart_trader.sleeve.capital import confidence_to_size_scale
from smart_trader.sleeve.models import SleeveDecision, SleeveId, SleeveState

log = structlog.get_logger(__name__)

MIN_CANDLES      = 60
MTF_THRESHOLD    = 0.10
SL_COOLDOWN      = 3
SL_COOLDOWN_HIGH = 8


class TacticalSleeve:
    """Evaluates a single tick and returns a SleeveDecision for the tactical sleeve."""

    def __init__(
        self,
        symbol:        str,
        signal_tf:     str   = "1h",
        trend_tf:      str   = "1d",
        mid_tf:        str   = "4h",
        min_confidence: float = 0.65,
        max_position_pct: float = 0.10,
        time_stop_hours: int  = 24,
        strategy_version: str = "v2",
    ) -> None:
        self.symbol          = symbol
        self.signal_tf       = signal_tf
        self.trend_tf        = trend_tf
        self.mid_tf          = mid_tf
        self._min_conf       = min_confidence
        self._time_stop_h    = time_stop_hours
        self._state          = SleeveState()
        self._log            = log.bind(sleeve="tactical", symbol=symbol)

        self._trend          = TrendEngine()
        self._signal         = SignalEngine(version=strategy_version,
                                            regime_routing=get_settings().hybrid_regime_routing)
        self._vol            = VolatilityAnalyzer()
        self._regime_adapter = RegimeParamAdapter()
        self._risk           = RiskManager(
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
        )

    def tick_down_cooldowns(self) -> None:
        for side in self._state.sl_cooldown:
            if self._state.sl_cooldown[side] > 0:
                self._state.sl_cooldown[side] -= 1

    def on_stop_loss(self, side: str) -> None:
        vol = self._state.vol_regime_at_entry.get(side, "")
        bars = SL_COOLDOWN_HIGH if vol in ("high", "crisis") else SL_COOLDOWN
        self._state.sl_cooldown[side] = bars
        self._log.info("sl_cooldown_set", side=side, bars=bars)

    def should_time_stop(self, opened_at: datetime) -> bool:
        """Return True if the position has exceeded the maximum hold duration."""
        now = datetime.now(timezone.utc)
        return (now - opened_at) >= timedelta(hours=self._time_stop_h)

    def evaluate(
        self,
        sig_df:    pd.DataFrame,
        trend_df:  pd.DataFrame,
        mid_df:    pd.DataFrame,
        portfolio: Portfolio,
        price:     float,
    ) -> SleeveDecision:
        """Run full signal pipeline for the tactical sleeve."""
        meta: dict = {"sleeve": "tactical", "price": price}

        # ── insufficient data ─────────────────────────────────────────────
        if len(sig_df) < MIN_CANDLES or len(trend_df) < MIN_CANDLES:
            meta["decision"] = "insufficient_data"
            return SleeveDecision(SleeveId.TACTICAL, None, None, portfolio, meta)

        # ── trend ─────────────────────────────────────────────────────────
        try:
            trend_state = self._trend.analyse(trend_df, self.symbol, self.trend_tf)
            meta["regime"] = str(trend_state.regime)
        except Exception as exc:
            self._log.warning("trend_error", error=str(exc))
            meta["decision"] = "trend_error"
            return SleeveDecision(SleeveId.TACTICAL, None, None, portfolio, meta)

        # ── mid-TF direction ──────────────────────────────────────────────
        mid_direction: Optional[float] = None
        if len(mid_df) >= MIN_CANDLES:
            try:
                mid_state    = self._trend.analyse(mid_df, self.symbol, self.mid_tf)
                mid_direction = mid_state.direction
                meta["mid_direction"] = round(mid_direction, 4)
            except Exception:
                pass

        # ── vol regime ────────────────────────────────────────────────────
        vol_state = None
        try:
            vol_state = self._vol.analyse(trend_df, sig_df)
            meta["vol_regime"] = str(vol_state.regime)
            meta["vol_state"]  = str(vol_state.state)
            self._state.cur_vol_regime = str(vol_state.regime)
            if vol_state.skip:
                meta["decision"] = "vol_skip"
                meta["vol_skip"] = True
                return SleeveDecision(SleeveId.TACTICAL, None, None, portfolio, meta)
        except Exception:
            pass

        # ── signals ───────────────────────────────────────────────────────
        _is_high_vol = self._state.cur_vol_regime in ("high", "crisis")
        _min_conf    = 0.75 if _is_high_vol else self._min_conf
        signals = self._signal.analyse(
            sig_df, symbol=self.symbol, timeframe=self.signal_tf,
            trend_state=trend_state, vol_state=vol_state,
            min_confidence=_min_conf,
        )
        if vol_state is not None and vol_state.preferred_sources and signals:
            signals = [
                s for s in signals
                if any(src in s.source for src in vol_state.preferred_sources)
            ]
        if not signals:
            meta["decision"] = "no_signals"
            return SleeveDecision(SleeveId.TACTICAL, None, None, portfolio, meta)

        best = signals[0]
        entry_side = "buy" if best.signal_type == "long" else "sell"
        meta["signals"] = [
            {
                "source": s.source, "signal_type": s.signal_type,
                "raw_score": round(s.raw_score, 4), "confidence": round(s.confidence, 4),
                "features": {k: round(v, 4) for k, v in s.features.items()},
            }
            for s in signals
        ]

        # ── cooldown ──────────────────────────────────────────────────────
        if self._state.sl_cooldown[entry_side] > 0:
            meta["decision"] = "cooldown"
            meta["decision_reason"] = f"{self._state.sl_cooldown[entry_side]} bars remaining"
            return SleeveDecision(SleeveId.TACTICAL, best, None, portfolio, meta)

        # ── MTF alignment ─────────────────────────────────────────────────
        if mid_direction is not None:
            if best.signal_type == "long" and mid_direction < -MTF_THRESHOLD:
                meta["decision"]        = "mtf_blocked"
                meta["decision_reason"] = f"4h direction={mid_direction:.3f}"
                return SleeveDecision(SleeveId.TACTICAL, best, None, portfolio, meta)
            if best.signal_type == "short" and mid_direction > MTF_THRESHOLD:
                meta["decision"]        = "mtf_blocked"
                meta["decision_reason"] = f"4h direction={mid_direction:.3f}"
                return SleeveDecision(SleeveId.TACTICAL, best, None, portfolio, meta)

        # ── risk evaluation ───────────────────────────────────────────────
        regime_params = self._regime_adapter.get(trend_state, vol_state)
        size_scale = confidence_to_size_scale(best.confidence) * regime_params.pos_cap_mult
        meta["pos_cap_mult"] = regime_params.pos_cap_mult
        decision = self._risk.evaluate(best, portfolio, price, sig_df, trend_df,
                                       size_scale=size_scale)
        if not decision.approved:
            meta["decision"]        = "risk_rejected"
            meta["decision_reason"] = "; ".join(decision.reasons)
            return SleeveDecision(SleeveId.TACTICAL, best, None, portfolio, meta)

        self._state.vol_regime_at_entry[entry_side] = self._state.cur_vol_regime
        meta["decision"]         = "enter"
        meta["entry_side"]       = entry_side
        meta["entry_confidence"] = round(best.confidence, 4)
        meta["size_scale"]       = size_scale
        return SleeveDecision(SleeveId.TACTICAL, best, decision, portfolio, meta)
