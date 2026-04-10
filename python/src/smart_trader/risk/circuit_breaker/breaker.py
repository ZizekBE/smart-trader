"""CircuitBreaker — hard-stop on daily loss or drawdown breach.

States:
  OPEN    — trading allowed
  TRIPPED — all new trades blocked until manually reset

Trips when:
  • daily_loss_pct  ≥ max_daily_loss_pct   (default 2 %)
  • drawdown_pct    ≥ max_drawdown_pct      (default 10 %)
"""
from __future__ import annotations

import structlog

from smart_trader.risk.models import CircuitBreakerResult, Portfolio

log = structlog.get_logger(__name__)


class CircuitBreaker:
    _OPEN    = "open"
    _TRIPPED = "tripped"

    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct:   float = 0.10,
    ) -> None:
        self._max_daily_loss = max_daily_loss_pct
        self._max_drawdown   = max_drawdown_pct
        self._state          = self._OPEN
        self._trip_reason:   str | None = None

    # ── public interface ───────────────────────────────────────

    @property
    def is_tripped(self) -> bool:
        return self._state == self._TRIPPED

    def check(self, portfolio: Portfolio) -> CircuitBreakerResult:
        """Evaluate portfolio state; trip if thresholds are breached."""
        daily_loss = portfolio.daily_loss_pct
        drawdown   = portfolio.drawdown_pct

        if not self.is_tripped:
            if daily_loss >= self._max_daily_loss:
                self._trip(
                    "daily_loss",
                    f"daily loss {daily_loss:.2%} ≥ limit {self._max_daily_loss:.2%}",
                )
            elif drawdown >= self._max_drawdown:
                self._trip(
                    "drawdown",
                    f"drawdown {drawdown:.2%} ≥ limit {self._max_drawdown:.2%}",
                )

        return CircuitBreakerResult(
            tripped=self.is_tripped,
            reason=self._trip_reason,
            daily_loss_pct=daily_loss,
            drawdown_pct=drawdown,
        )

    def reset(self) -> None:
        """Manually re-open the breaker (e.g., at start of new trading day)."""
        if self.is_tripped:
            log.warning("circuit_breaker_reset", prev_reason=self._trip_reason)
        self._state       = self._OPEN
        self._trip_reason = None

    # ── private ────────────────────────────────────────────────

    def _trip(self, reason_key: str, detail: str) -> None:
        self._state       = self._TRIPPED
        self._trip_reason = reason_key
        log.error("circuit_breaker_tripped", reason=reason_key, detail=detail)
