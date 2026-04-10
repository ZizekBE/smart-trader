"""PerformanceReport — ASCII + JSON report from PerformanceMetrics / BacktestResult."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union

from smart_trader.analysis.metrics.calculator import PerformanceMetrics


def _fmt(v: float, decimals: int = 2) -> str:
    if math.isinf(v):
        return "∞"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def _pct(v: float) -> str:
    return f"{v:.2%}"


class PerformanceReport:
    """Build human-readable and machine-readable reports."""

    def __init__(
        self,
        metrics: PerformanceMetrics,
        title: str = "Performance Report",
        start_at: Optional[datetime] = None,
        end_at:   Optional[datetime] = None,
    ) -> None:
        self.metrics  = metrics
        self.title    = title
        self.start_at = start_at
        self.end_at   = end_at

    # ── ASCII ─────────────────────────────────────────────────────────────────

    def ascii(self) -> str:
        m   = self.metrics
        sep = "─" * 54

        lines = [
            "",
            f"  {self.title}",
        ]
        if self.start_at or self.end_at:
            s = self.start_at.strftime("%Y-%m-%d") if self.start_at else "?"
            e = self.end_at.strftime("%Y-%m-%d")   if self.end_at   else "?"
            lines.append(f"  Period : {s}  →  {e}")
        lines += [
            sep,
            "  OVERVIEW",
            f"    Total trades      : {m.total_trades}",
            f"    Winning / Losing  : {m.winning_trades} / {m.losing_trades}",
            f"    Win rate          : {_pct(m.win_rate)}",
            sep,
            "  P&L",
            f"    Total P&L         : ${_fmt(m.total_pnl)}  ({_pct(m.total_pnl_pct)})",
            f"    Gross profit      : ${_fmt(m.gross_profit)}",
            f"    Gross loss        : ${_fmt(m.gross_loss)}",
            f"    Profit factor     : {_fmt(m.profit_factor, 3)}",
            f"    Expectancy        : ${_fmt(m.expectancy, 4)} / trade",
            f"    Avg win           : ${_fmt(m.avg_win, 4)}",
            f"    Avg loss          : ${_fmt(m.avg_loss, 4)}",
            f"    Largest win       : ${_fmt(m.largest_win, 4)}",
            f"    Largest loss      : ${_fmt(m.largest_loss, 4)}",
            sep,
            "  RISK",
            f"    Sharpe ratio      : {_fmt(m.sharpe_ratio, 4)}",
            f"    Sortino ratio     : {_fmt(m.sortino_ratio, 4)}",
            f"    Calmar ratio      : {_fmt(m.calmar_ratio, 4)}",
            f"    Max drawdown      : {_pct(m.max_drawdown_pct)}  (${_fmt(m.max_drawdown_usd)})",
            sep,
            "  COSTS",
            f"    Total fees        : ${_fmt(m.total_fees, 6)}",
            f"    Total slippage    : ${_fmt(m.total_slippage, 6)}",
            f"    Fee drag          : {_pct(m.fee_drag_pct)}",
            sep,
            "  DURATION",
            f"    Avg trade dur.    : {_fmt(m.avg_trade_duration_hours)} h",
            sep,
            "  CAPITAL",
            f"    Initial capital   : ${_fmt(m.initial_capital)}",
            f"    Final capital     : ${_fmt(m.final_capital)}",
            sep,
            "",
        ]
        return "\n".join(lines)

    # ── JSON ──────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        m = self.metrics
        d = {
            "title":    self.title,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at":   self.end_at.isoformat()   if self.end_at   else None,
            "overview": {
                "total_trades":   m.total_trades,
                "winning_trades": m.winning_trades,
                "losing_trades":  m.losing_trades,
                "win_rate":       m.win_rate,
            },
            "pnl": {
                "total_pnl":     m.total_pnl,
                "total_pnl_pct": m.total_pnl_pct,
                "gross_profit":  m.gross_profit,
                "gross_loss":    m.gross_loss,
                "profit_factor": m.profit_factor if math.isfinite(m.profit_factor) else None,
                "expectancy":    m.expectancy,
                "avg_win":       m.avg_win,
                "avg_loss":      m.avg_loss,
                "largest_win":   m.largest_win,
                "largest_loss":  m.largest_loss,
            },
            "risk": {
                "sharpe_ratio":     m.sharpe_ratio,
                "sortino_ratio":    m.sortino_ratio if math.isfinite(m.sortino_ratio) else None,
                "calmar_ratio":     m.calmar_ratio,
                "max_drawdown_pct": m.max_drawdown_pct,
                "max_drawdown_usd": m.max_drawdown_usd,
            },
            "costs": {
                "total_fees":     m.total_fees,
                "total_slippage": m.total_slippage,
                "fee_drag_pct":   m.fee_drag_pct,
            },
            "duration": {
                "avg_trade_duration_hours": m.avg_trade_duration_hours,
            },
            "capital": {
                "initial_capital": m.initial_capital,
                "final_capital":   m.final_capital,
            },
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return self.ascii()
