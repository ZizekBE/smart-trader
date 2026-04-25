"""CoreSleeve — long-term trend-following positions.

Signal timeframe : 4h   (configurable)
Trend timeframe  : 1d
Confidence floor : 0.70
Stop style       : wide ATR-based (3.0×ATR default) — holds through volatility
Take-profit      : disabled; trailing stop used instead (from BacktestConfig logic)
Max positions    : 2 per symbol-side pair
Time stop        : none (positions held until stop or trend reversal)
"""
from __future__ import annotations

import structlog
from typing import Optional

import pandas as pd

from smart_trader.core.settings import get_settings
from smart_trader.risk.manager import RiskManager
from smart_trader.risk.models import Portfolio, RiskDecision
from smart_trader.strategy.adaptive_params import RegimeParamAdapter
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import MarketRegime
from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer
from smart_trader.sleeve.capital import confidence_to_size_scale
from smart_trader.sleeve.models import SleeveDecision, SleeveId, SleeveState

log = structlog.get_logger(__name__)

MIN_CANDLES   = 60
MTF_THRESHOLD = 0.10
SL_COOLDOWN   = 4     # longer cooldown after core-sleeve stop

_SHORT_ENTRY_REGIMES = frozenset({
    MarketRegime.BEAR_TRENDING,
    MarketRegime.BEAR_RANGING,
    MarketRegime.DISTRIBUTION,
})


class CoreSleeve:
    """Evaluates a single tick and returns a SleeveDecision for the core sleeve."""

    def __init__(
        self,
        symbol:        str,
        signal_tf:     str   = "4h",
        trend_tf:      str   = "1d",
        min_confidence: float = 0.70,
        max_position_pct: float = 0.30,
        atr_mult:      float = 3.0,
        rr_ratio:      float = 3.0,
        strategy_version: str = "v2",
    ) -> None:
        self.symbol        = symbol
        self.signal_tf     = signal_tf
        self.trend_tf      = trend_tf
        self._min_conf     = min_confidence
        self._state        = SleeveState()
        self._log          = log.bind(sleeve="core", symbol=symbol)

        self._trend   = TrendEngine()
        self._signal  = SignalEngine(version=strategy_version,
                                     regime_routing=get_settings().hybrid_regime_routing)
        self._vol     = VolatilityAnalyzer()
        self._regime_adapter = RegimeParamAdapter()
        self._risk   = RiskManager(
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
            atr_mult=atr_mult,
            rr_ratio=rr_ratio,
        )

    def tick_down_cooldowns(self) -> None:
        for side in self._state.sl_cooldown:
            if self._state.sl_cooldown[side] > 0:
                self._state.sl_cooldown[side] -= 1

    def on_stop_loss(self, side: str) -> None:
        """Called by the hybrid loop when a core trade is stopped out."""
        self._state.sl_cooldown[side] = SL_COOLDOWN
        self._log.info("sl_cooldown_set", side=side, bars=SL_COOLDOWN)

    def evaluate(
        self,
        sig_df:    pd.DataFrame,
        trend_df:  pd.DataFrame,
        portfolio: Portfolio,
        price:     float,
    ) -> SleeveDecision:
        """Run signal detection and risk evaluation for the core sleeve.

        Returns a SleeveDecision with signal + decision (or None if no trade).
        """
        meta: dict = {"sleeve": "core", "price": price}

        # ── insufficient data ─────────────────────────────────────────────
        if len(sig_df) < MIN_CANDLES or len(trend_df) < MIN_CANDLES:
            meta["decision"] = "insufficient_data"
            return SleeveDecision(SleeveId.CORE, None, None, portfolio, meta)

        # ── trend analysis ────────────────────────────────────────────────
        try:
            trend_state = self._trend.analyse(trend_df, self.symbol, self.trend_tf)
            meta["regime"] = str(trend_state.regime)
        except Exception as exc:
            self._log.warning("trend_error", error=str(exc))
            meta["decision"] = "trend_error"
            return SleeveDecision(SleeveId.CORE, None, None, portfolio, meta)

        # ── vol regime ────────────────────────────────────────────────────
        vol_state: Optional[VolatilityState] = None
        try:
            vol_state = self._vol.analyse(trend_df, sig_df)
            meta["vol_regime"] = str(vol_state.regime)
            self._state.cur_vol_regime = str(vol_state.regime)
            if vol_state.skip:
                meta["decision"] = "vol_skip"
                return SleeveDecision(SleeveId.CORE, None, None, portfolio, meta)
        except Exception:
            pass

        # ── signal detection ──────────────────────────────────────────────
        signals = self._signal.analyse(
            sig_df, symbol=self.symbol, timeframe=self.signal_tf,
            trend_state=trend_state, vol_state=vol_state,
            min_confidence=self._min_conf,
        )
        # vol-source filter
        if vol_state is not None and vol_state.preferred_sources and signals:
            signals = [
                s for s in signals
                if any(src in s.source for src in vol_state.preferred_sources)
            ]
        if not signals:
            meta["decision"] = "no_signals"
            return SleeveDecision(SleeveId.CORE, None, None, portfolio, meta)

        best = signals[0]
        entry_side = "buy" if best.signal_type == "long" else "sell"

        # ── short-entry regime gate ───────────────────────────────────────
        if entry_side == "sell" and trend_state.regime not in _SHORT_ENTRY_REGIMES:
            meta["decision"] = "short_regime_blocked"
            meta["decision_reason"] = f"regime={trend_state.regime} not in short-allowed set"
            return SleeveDecision(SleeveId.CORE, best, None, portfolio, meta)

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
            return SleeveDecision(SleeveId.CORE, best, None, portfolio, meta)

        # ── risk evaluation ───────────────────────────────────────────────
        regime_params = self._regime_adapter.get(trend_state, vol_state)
        size_scale = confidence_to_size_scale(best.confidence) * regime_params.pos_cap_mult
        meta["pos_cap_mult"] = regime_params.pos_cap_mult
        decision = self._risk.evaluate(best, portfolio, price, sig_df, trend_df,
                                       size_scale=size_scale)
        if not decision.approved:
            meta["decision"] = "risk_rejected"
            meta["decision_reason"] = "; ".join(decision.reasons)
            return SleeveDecision(SleeveId.CORE, best, None, portfolio, meta)

        # Record vol regime for adaptive cooldown later
        self._state.vol_regime_at_entry[entry_side] = self._state.cur_vol_regime

        meta["decision"]         = "enter"
        meta["entry_side"]       = entry_side
        meta["entry_confidence"] = round(best.confidence, 4)
        meta["size_scale"]       = size_scale
        return SleeveDecision(SleeveId.CORE, best, decision, portfolio, meta)
