"""CorrelationGuard — rolling cross-asset correlation check for EPIC-MULTI.

Computes the 30-day rolling Pearson correlation between two symbols using
daily close prices. If correlation exceeds the threshold, both symbols'
position cap multipliers are reduced to avoid doubling correlated risk.

Usage::
    guard = CorrelationGuard()
    mult  = guard.position_cap_mult(eth_df_1d, btc_df_1d)
    # mult is 1.0 (normal) or 0.80 (correlated — reduce exposure 20%)
"""
from __future__ import annotations

import structlog

import pandas as pd

log = structlog.get_logger(__name__)

_CORR_WINDOW    = 30   # calendar days of daily closes
_CORR_THRESHOLD = 0.80
_CORR_REDUCTION = 0.20  # reduce pos cap by this fraction when correlated


class CorrelationGuard:
    """Stateless helper — computes correlation and returns a cap multiplier."""

    def __init__(
        self,
        window:    int   = _CORR_WINDOW,
        threshold: float = _CORR_THRESHOLD,
        reduction: float = _CORR_REDUCTION,
    ) -> None:
        self._window    = window
        self._threshold = threshold
        self._reduction = reduction

    def position_cap_mult(
        self,
        df_a: pd.DataFrame,  # daily OHLCV for symbol A
        df_b: pd.DataFrame,  # daily OHLCV for symbol B
    ) -> float:
        """Return position-cap multiplier: 1.0 normally, (1-reduction) if correlated."""
        try:
            corr = self._compute(df_a, df_b)
            if corr is None:
                return 1.0
            log.info("correlation_check", corr=round(corr, 4),
                     threshold=self._threshold, high=corr >= self._threshold)
            if corr >= self._threshold:
                return 1.0 - self._reduction
        except Exception as exc:
            log.warning("correlation_error", error=str(exc))
        return 1.0

    def _compute(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame
    ) -> float | None:
        if df_a.empty or df_b.empty:
            return None
        close_a = df_a["close"].astype(float).tail(self._window)
        close_b = df_b["close"].astype(float)
        # align on index (timestamps may differ slightly)
        aligned = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
        if len(aligned) < max(10, self._window // 2):
            return None
        return float(aligned["a"].corr(aligned["b"]))
