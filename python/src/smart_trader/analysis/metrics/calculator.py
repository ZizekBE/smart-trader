"""MetricsCalculator — compute trading performance metrics from closed trades.

Metrics produced:
  P&L       total_pnl, total_pnl_pct, gross_profit, gross_loss, profit_factor
  Per-trade  win_rate, avg_win, avg_loss, largest_win, largest_loss, expectancy
  Risk       sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
  Costs      total_fees, total_slippage, fee_drag_pct
  Duration   avg_trade_duration_hours
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass
class TradeResult:
    """Minimal closed-trade record for metrics calculation."""
    pnl:        float
    pnl_pct:    float
    opened_at:  datetime
    closed_at:  datetime
    fees:       float
    slippage:   float
    side:       str   # 'buy' | 'sell'


@dataclass
class PerformanceMetrics:
    # ── overview ──────────────────────────────────────────────
    total_trades:   int
    winning_trades: int
    losing_trades:  int
    win_rate:       float   # [0, 1]

    # ── P&L ───────────────────────────────────────────────────
    total_pnl:      float   # USDT
    total_pnl_pct:  float   # vs initial capital
    gross_profit:   float
    gross_loss:     float   # positive number
    profit_factor:  float   # gross_profit / gross_loss  (∞ if no losses)
    avg_win:        float
    avg_loss:       float   # negative number
    largest_win:    float
    largest_loss:   float   # negative number
    expectancy:     float   # expected USDT per trade

    # ── risk ──────────────────────────────────────────────────
    sharpe_ratio:     float
    sortino_ratio:    float
    calmar_ratio:     float
    max_drawdown_pct: float   # [0, 1]
    max_drawdown_usd: float

    # ── costs ─────────────────────────────────────────────────
    total_fees:     float
    total_slippage: float
    fee_drag_pct:   float   # fees as % of initial capital

    # ── duration ──────────────────────────────────────────────
    avg_trade_duration_hours: float

    # ── capital ───────────────────────────────────────────────
    initial_capital: float
    final_capital:   float

    # ── annualised ────────────────────────────────────────────
    annualized_return: float  # CAGR over actual backtest period

    def __str__(self) -> str:
        return (
            f"Trades={self.total_trades}  WR={self.win_rate:.1%}  "
            f"PnL=${self.total_pnl:.2f} ({self.total_pnl_pct:.2%})  "
            f"Sharpe={self.sharpe_ratio:.2f}  MDD={self.max_drawdown_pct:.1%}"
        )


_ANNUAL_FACTOR = 252   # trading days


class MetricsCalculator:
    """Compute PerformanceMetrics from a list of closed TradeResults."""

    def calculate(
        self,
        trades:          Sequence[TradeResult],
        initial_capital: float = 10_000.0,
    ) -> PerformanceMetrics:
        n = len(trades)

        if n == 0:
            return self._empty(initial_capital)

        # sort by close time for equity curve
        sorted_trades = sorted(trades, key=lambda t: t.closed_at)
        pnls = [t.pnl for t in sorted_trades]

        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_pnl    = sum(pnls)
        final_capital = initial_capital + total_pnl

        avg_win  = (sum(wins)   / len(wins))   if wins   else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        largest_win  = max(pnls) if pnls else 0.0
        largest_loss = min(pnls) if pnls else 0.0

        win_rate   = len(wins) / n
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # ── equity curve & drawdown ────────────────────────────
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        equity = equity[1:]   # drop initial seed

        max_dd_pct, max_dd_usd = self._max_drawdown(equity, initial_capital)

        # ── daily returns for Sharpe / Sortino ────────────────
        daily_rets = self._daily_returns(sorted_trades, initial_capital)
        sharpe  = self._sharpe(daily_rets)
        sortino = self._sortino(daily_rets)
        calmar  = (total_pnl / initial_capital) / max_dd_pct if max_dd_pct > 0 else 0.0

        # ── costs ──────────────────────────────────────────────
        total_fees     = sum(t.fees     for t in sorted_trades)
        total_slippage = sum(t.slippage for t in sorted_trades)
        fee_drag_pct   = total_fees / initial_capital

        # ── duration ───────────────────────────────────────────
        durations_h = [
            (t.closed_at - t.opened_at).total_seconds() / 3600
            for t in sorted_trades
            if t.closed_at > t.opened_at
        ]
        avg_duration = sum(durations_h) / len(durations_h) if durations_h else 0.0

        # ── annualised return (CAGR) ───────────────────────────
        period_days = (
            (sorted_trades[-1].closed_at - sorted_trades[0].opened_at).total_seconds() / 86400
        )
        if period_days >= 1:
            ann_return = (1 + total_pnl / initial_capital) ** (365 / period_days) - 1
        else:
            ann_return = 0.0

        return PerformanceMetrics(
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=round(total_pnl, 4),
            total_pnl_pct=total_pnl / initial_capital,
            gross_profit=round(gross_profit, 4),
            gross_loss=round(gross_loss, 4),
            profit_factor=round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            largest_win=round(largest_win, 4),
            largest_loss=round(largest_loss, 4),
            expectancy=round(expectancy, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown_pct=round(max_dd_pct, 6),
            max_drawdown_usd=round(max_dd_usd, 4),
            total_fees=round(total_fees, 6),
            total_slippage=round(total_slippage, 6),
            fee_drag_pct=round(fee_drag_pct, 6),
            avg_trade_duration_hours=round(avg_duration, 2),
            initial_capital=initial_capital,
            final_capital=round(final_capital, 4),
            annualized_return=round(ann_return, 6),
        )

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(equity: list[float], initial: float) -> tuple[float, float]:
        peak = initial
        max_dd_pct = 0.0
        max_dd_usd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd_usd = peak - e
            dd_pct = dd_usd / peak if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_usd = dd_usd
        return max_dd_pct, max_dd_usd

    @staticmethod
    def _daily_returns(
        trades: list[TradeResult], initial_capital: float
    ) -> list[float]:
        """Approximate daily returns by grouping trade P&L by calendar date."""
        from collections import defaultdict
        daily: dict = defaultdict(float)
        for t in trades:
            daily[t.closed_at.date()] += t.pnl

        if not daily:
            return []

        dates = sorted(daily)
        capital = initial_capital
        rets = []
        for d in dates:
            pnl  = daily[d]
            ret  = pnl / capital
            rets.append(ret)
            capital += pnl
        return rets

    @staticmethod
    def _sharpe(daily_rets: list[float], rf: float = 0.0) -> float:
        if len(daily_rets) < 2:
            return 0.0
        mean = sum(daily_rets) / len(daily_rets) - rf
        var  = sum((r - mean - rf) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
        std  = math.sqrt(var)
        return (mean / std * math.sqrt(_ANNUAL_FACTOR)) if std > 0 else 0.0

    @staticmethod
    def _sortino(daily_rets: list[float], rf: float = 0.0) -> float:
        if len(daily_rets) < 2:
            return 0.0
        mean      = sum(daily_rets) / len(daily_rets) - rf
        neg_rets  = [r for r in daily_rets if r < rf]
        if not neg_rets:
            return float("inf")
        downside_var = sum(r ** 2 for r in neg_rets) / len(neg_rets)
        downside_std = math.sqrt(downside_var)
        return (mean / downside_std * math.sqrt(_ANNUAL_FACTOR)) if downside_std > 0 else 0.0

    @staticmethod
    def _empty(initial_capital: float) -> PerformanceMetrics:
        z = 0.0
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=z,
            total_pnl=z, total_pnl_pct=z, gross_profit=z, gross_loss=z,
            profit_factor=z, avg_win=z, avg_loss=z, largest_win=z,
            largest_loss=z, expectancy=z, sharpe_ratio=z, sortino_ratio=z,
            calmar_ratio=z, max_drawdown_pct=z, max_drawdown_usd=z,
            total_fees=z, total_slippage=z, fee_drag_pct=z,
            avg_trade_duration_hours=z,
            initial_capital=initial_capital, final_capital=initial_capital,
            annualized_return=z,
        )
