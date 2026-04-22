"""BacktestEngine — in-memory walk-forward backtest.

Replays the full signal → risk → execution pipeline on historical candles
without touching the database or any live exchange.

Design
──────
• Feed OHLCV candles chronologically (one bar at a time).
• On each bar: run TrendEngine + SignalEngine to detect signals.
• Run RiskManager.evaluate() against a virtual Portfolio.
• Simulate fills (slippage + fee) identically to paper OrderManager.
• Check SL/TP/trailing-stop against bar's high/low (realistic fill).
• Collect BacktestTrade records; compute PerformanceMetrics at the end.

Signal-quality guards
──────────────────────
• SL cooldown      : skip N bars after stop-loss (adaptive to vol regime)
• Consecutive SL   : double cooldown after N stops in same direction
• MTF filter       : signal must align with mid-timeframe trend
• Vol regime gate  : skip new entries during CRISIS / HIGH+SPIKE
• High-vol conf.   : raised confidence threshold in HIGH/CRISIS regime

Exit management
───────────────
• Breakeven stop   : move stop to entry price at cfg.breakeven_trigger_pct of TP
• Trailing stop    : ATR-based trail activates at cfg.trailing_trigger_pct of TP
• Trailing exit does NOT trigger stop-loss cooldown (profit was locked)

Pullback entry
──────────────
• Instead of filling at bar close, set a limit 0.3×ATR better than close
• Wait up to cfg.pullback_max_bars for the limit to be hit
• If not hit: skip the trade (avoids chasing strong momentum)
• SL/TP levels are preserved from the signal bar (based on close price),
  so a pullback fill improves effective R:R at no extra risk
"""
from __future__ import annotations

import bisect
import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from smart_trader.analysis.metrics.calculator import MetricsCalculator, TradeResult, PerformanceMetrics
from smart_trader.risk.manager import RiskManager
from smart_trader.risk.models import OpenPosition, Portfolio
from smart_trader.risk.position.sizer import PositionSizer
from smart_trader.strategy.signals.engine import SignalEngine
from smart_trader.strategy.trend.engine import TrendEngine
from smart_trader.strategy.trend.regime import MarketRegime, StrategyMode
from smart_trader.strategy.volatility.analyzer import VolatilityAnalyzer

# ── constants (mirrors paper OrderManager) ────────────────────────────────────
FEE_RATE      = 0.001   # 0.10 %
SLIPPAGE_RATE = 0.0005  # 0.05 %
MIN_CANDLES   = 60      # minimum candles before signals fire

# Regimes where SL cooldown is reduced (trend acts as re-entry confirmation)
_TRENDING_REGIMES = frozenset({MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING})
# Regimes where SL cooldown is extended (choppy/distribution = high whipsaw risk)
_CAUTIOUS_REGIMES = frozenset({MarketRegime.DISTRIBUTION, MarketRegime.BEAR_RANGING})


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A single completed backtest trade."""
    trade_id:    uuid.UUID
    symbol:      str
    side:        str              # 'buy' | 'sell'
    entry_time:  datetime
    exit_time:   datetime
    entry_price: float
    exit_price:  float
    quantity:    float
    pnl:         float
    pnl_pct:     float
    fees:        float
    slippage:    float
    exit_reason: str             # 'stop_loss' | 'breakeven' | 'trailing_stop' | 'take_profit' | 'end_of_data'
    stop_price:  float
    tp_price:    float
    confidence:  float
    regime:      str


@dataclass
class BacktestConfig:
    symbol:           str   = "BTC/USDT"
    timeframe:        str   = "1h"
    trend_timeframe:  str   = "1d"
    initial_capital:  float = 10_000.0
    min_confidence:   float = 0.65
    max_position_pct: float = 0.05
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct:   float = 0.10

    # signal quality controls
    sl_cooldown_bars:          int   = 3     # bars to skip after stop-loss (normal vol)
    sl_cooldown_high_vol_bars: int   = 8     # bars to skip after stop-loss in HIGH/CRISIS vol
    min_confidence_high_vol:   float = 0.75  # raised threshold in HIGH/CRISIS vol
    require_mtf:               bool  = True  # require mid-timeframe trend alignment
    consecutive_sl_max:        int   = 2     # consecutive stops before cooldown doubles

    # breakeven & trailing stop
    breakeven_trigger_pct:  float = 0.50   # activate BE at this fraction of TP distance gained
    trailing_trigger_pct:   float = 0.75   # activate trailing stop at this fraction of TP distance
    trailing_atr_mult:      float = 1.5    # trail distance = trailing_atr_mult × ATR

    # pullback entry
    pullback_entry:    bool  = True   # wait for pullback instead of filling at bar close
    pullback_atr_frac: float = 0.30   # limit = close ± pullback_atr_frac × ATR from close
    pullback_max_bars: int   = 3      # cancel pending entry after this many unfilled bars

    # signal strategy version + Phase 2.3 regime routing
    strategy_version: str  = "v2"     # set to "v1" to run baseline for comparison
    regime_routing:   bool = False    # True = route strategy preset by MarketRegime

    # risk param overrides (None = use PositionSizer mode defaults)
    atr_mult:   Optional[float] = None   # stop-loss ATR multiplier
    rr_ratio:   Optional[float] = None   # base reward:risk ratio
    kelly_frac: Optional[float] = None   # fractional Kelly multiplier

    # session filter — skip new entries outside active trading hours
    session_filter:    bool = False
    session_start_utc: int  = 7    # inclusive (EU open)
    session_end_utc:   int  = 22   # exclusive (US late session close)

    # partial profit taking — close a fraction at N×R, lock breakeven on remainder
    partial_tp:     bool  = False
    partial_tp_r:   float = 1.0    # take partial profit at N×R (1R = one SL distance)
    partial_tp_pct: float = 0.50   # fraction of position to close (0.50 = 50%)

    # bear market filter — pause new long entries when trend direction has been
    # negative for N consecutive days (uses continuous direction, not discrete regime,
    # so it catches BEAR_RANGING and DISTRIBUTION as well as BEAR_TRENDING)
    bear_filter:           bool  = True    # enabled by default
    bear_filter_bars:      int   = 3       # consecutive bearish days to trigger pause
    bear_filter_threshold: float = -0.15   # direction threshold (< this = bearish day)
    bear_filter_long_only: bool  = True    # only block longs (allow shorts)


@dataclass
class BacktestResult:
    config:   BacktestConfig
    trades:   list[BacktestTrade]
    metrics:  PerformanceMetrics
    start_at: Optional[datetime] = None
    end_at:   Optional[datetime] = None


# ── open-position tracker (in-memory only) ────────────────────────────────────

@dataclass
class _OpenPos:
    trade_id:    uuid.UUID
    symbol:      str
    side:        str
    entry_price: float
    quantity:    float
    stop_price:  float        # original (initial) stop
    tp_price:    float
    entry_time:  datetime
    entry_fee:   float
    entry_slip:  float
    confidence:  float
    regime:      str
    trail_atr:   float = 0.0  # ATR at entry — used for trailing stop distance
    # live exit management (mutated per bar)
    active_stop:         float = 0.0
    best_price:          float = 0.0   # best price seen (max high for long, min low for short)
    breakeven_activated: bool  = False
    trailing_active:     bool  = False
    partial_closed:      bool  = False  # True once partial TP has been taken

    def __post_init__(self) -> None:
        if self.active_stop == 0.0:
            self.active_stop = self.stop_price
        if self.best_price == 0.0:
            self.best_price = self.entry_price


# ── pending entry (pullback limit order) ─────────────────────────────────────

@dataclass
class _PendingEntry:
    """Limit order waiting for a pullback fill."""
    entry_side:  str
    limit_price: float    # entry price we're waiting for
    quantity:    float
    stop_price:  float    # SL level (calculated from signal-bar close)
    tp_price:    float    # TP level (calculated from signal-bar close)
    confidence:  float
    regime:      str
    trail_atr:   float    # ATR at signal bar (for trailing stop)
    bars_left:   int      # countdown to expiry


# ── engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """Stateless once run() is called; construct fresh per backtest."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        signal_candles: pd.DataFrame,
        trend_candles:  pd.DataFrame | None = None,
        mid_candles:    pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        signal_candles:
            DataFrame of OHLCV for the signal timeframe (e.g. 1 h).
            Required columns: open, high, low, close, volume.
            Index or 'time' column must be datetime.
        trend_candles:
            Optional OHLCV for the trend timeframe (e.g. 1 d).
            When omitted, the signal candles are also used for trend.
        mid_candles:
            Optional OHLCV for an intermediate timeframe (e.g. 4 h).
            When provided and cfg.require_mtf=True, signals must align
            with the mid-timeframe trend direction.
        """
        sig_df   = self._normalise(signal_candles)
        trend_df = self._normalise(trend_candles) if trend_candles is not None else sig_df
        mid_df   = self._normalise(mid_candles)   if mid_candles   is not None else None

        trend_eng    = TrendEngine()
        signal_eng   = SignalEngine(version=self.cfg.strategy_version,
                                    regime_routing=self.cfg.regime_routing)
        vol_analyzer = VolatilityAnalyzer()

        # ── pre-compute trend / mid states (cache by bar time) ────────────────
        # TrendEngine is O(window) per call.  Instead of running it once per
        # signal bar (n_signal), we run it once per trend/mid bar (n_trend,
        # n_mid) and cache the result.  Lookup is O(log n) via bisect.
        _trend_times:   list = []    # sorted timestamps of valid cached states
        _trend_df_rows: list = []    # corresponding row indices in trend_df
        _trend_cache:   dict = {}    # bar_time → TrendState
        # Start at MIN_CANDLES-1 so the window at index j contains exactly
        # MIN_CANDLES rows — matches the original `len(window) < MIN_CANDLES`
        # guard (strict less-than, so equal is allowed).
        for _j in range(MIN_CANDLES - 1, len(trend_df)):
            _tw = trend_df.iloc[max(0, _j - 199): _j + 1]
            try:
                _ts = trend_eng.analyse(_tw, self.cfg.symbol, self.cfg.trend_timeframe)
                _t  = trend_df.index[_j]
                _trend_times.append(_t)
                _trend_df_rows.append(_j)
                _trend_cache[_t] = _ts
            except ValueError:
                pass

        _mid_times:  list = []
        _mid_cache:  dict = {}     # bar_time → direction float
        if mid_df is not None and self.cfg.require_mtf:
            for _j in range(MIN_CANDLES - 1, len(mid_df)):
                _mw = mid_df.iloc[max(0, _j - 199): _j + 1]
                try:
                    _ms = trend_eng.analyse(_mw, self.cfg.symbol, "mid")
                    _mt = mid_df.index[_j]
                    _mid_times.append(_mt)
                    _mid_cache[_mt] = _ms.direction
                except ValueError:
                    pass

        risk_mgr     = RiskManager(
            min_confidence=self.cfg.min_confidence,
            max_position_pct=self.cfg.max_position_pct,
            max_daily_loss_pct=self.cfg.max_daily_loss_pct,
            max_drawdown_pct=self.cfg.max_drawdown_pct,
            atr_mult=self.cfg.atr_mult,
            rr_ratio=self.cfg.rr_ratio,
            kelly_frac=self.cfg.kelly_frac,
        )

        capital     = self.cfg.initial_capital
        open_pos:   Optional[_OpenPos]     = None
        pending:    Optional[_PendingEntry] = None
        closed_trades: list[BacktestTrade] = []

        sl_cooldown:     dict[str, int] = {"buy": 0, "sell": 0}
        sl_cooldown_vol: dict[str, str] = {"buy": "", "sell": ""}
        consecutive_sl:  dict[str, int] = {"buy": 0, "sell": 0}
        last_trend_state = None   # used for regime-scaled cooldown at SL set-time
        consecutive_bear: int = 0       # consecutive BEAR_TRENDING *days* (deduped by date)
        _last_bear_date:  Optional[datetime] = None  # last date we incremented the counter

        for i in range(MIN_CANDLES, len(sig_df)):
            bar         = sig_df.iloc[i]
            bar_time    = bar.name if isinstance(bar.name, datetime) else bar["time"]
            close_price = float(bar["close"])

            # tick down cooldown counters
            for s in sl_cooldown:
                if sl_cooldown[s] > 0:
                    sl_cooldown[s] -= 1

            # ── 1. check exits on open position ───────────────────────────
            if open_pos is not None:
                # 1a. partial profit taking (runs before full exit check)
                if self.cfg.partial_tp and not open_pos.partial_closed:
                    pt = self._check_partial(
                        open_pos, bar, bar_time,
                        self.cfg.partial_tp_r, self.cfg.partial_tp_pct,
                    )
                    if pt is not None:
                        closed_trades.append(pt)
                        capital += pt.pnl

                # 1b. full exit check
                ct = self._check_exit(
                    open_pos, bar, bar_time,
                    breakeven_trigger_pct=self.cfg.breakeven_trigger_pct,
                    trailing_trigger_pct=self.cfg.trailing_trigger_pct,
                    trailing_atr_mult=self.cfg.trailing_atr_mult,
                )
                if ct is not None:
                    closed_trades.append(ct)
                    capital += ct.pnl
                    side = open_pos.side
                    if ct.exit_reason == "stop_loss":
                        consecutive_sl[side] += 1
                        _vol_reg = sl_cooldown_vol.get(side, "")
                        base_cd  = (
                            self.cfg.sl_cooldown_high_vol_bars
                            if _vol_reg in ("high", "crisis")
                            else self.cfg.sl_cooldown_bars
                        )
                        if consecutive_sl[side] >= self.cfg.consecutive_sl_max:
                            base_cd *= 2
                        # Regime-scaled cooldown: in trending markets re-entry is
                        # valid sooner; in choppy/distribution stay out longer.
                        if last_trend_state is not None:
                            if last_trend_state.regime in _TRENDING_REGIMES:
                                base_cd = max(1, base_cd - 2)
                            elif last_trend_state.regime in _CAUTIOUS_REGIMES:
                                base_cd = base_cd + 2
                        sl_cooldown[side] = base_cd
                    else:
                        # TP, breakeven, or trailing_stop all reset the streak
                        consecutive_sl[side] = 0
                    open_pos = None
                    risk_mgr.reset_breaker()

            # ── 2. check pending pullback entry ───────────────────────────
            if pending is not None and open_pos is None:
                bar_low  = float(bar["low"])
                bar_high = float(bar["high"])
                filled   = (
                    (pending.entry_side == "buy"  and bar_low  <= pending.limit_price) or
                    (pending.entry_side == "sell" and bar_high >= pending.limit_price)
                )
                if filled:
                    fill     = pending.limit_price   # limit fills at exact limit price
                    fee      = fill * pending.quantity * FEE_RATE
                    capital -= fee
                    _cur_vol = sl_cooldown_vol.get(pending.entry_side, "")
                    open_pos = _OpenPos(
                        trade_id=uuid.uuid4(),
                        symbol=self.cfg.symbol,
                        side=pending.entry_side,
                        entry_price=fill,
                        quantity=pending.quantity,
                        stop_price=pending.stop_price,
                        tp_price=pending.tp_price,
                        entry_time=bar_time,
                        entry_fee=fee,
                        entry_slip=0.0,   # no slippage on limit order
                        confidence=pending.confidence,
                        regime=pending.regime,
                        trail_atr=pending.trail_atr,
                    )
                    pending = None
                else:
                    pending.bars_left -= 1
                    if pending.bars_left <= 0:
                        pending = None   # expired — skip this signal

            # ── 3. skip bar if position or pending entry exists ────────────
            if open_pos is not None or pending is not None:
                continue

            # ── 3b. session filter — skip new entries in low-activity hours ─
            if self.cfg.session_filter:
                _hour = bar_time.hour if hasattr(bar_time, "hour") else pd.Timestamp(bar_time).hour
                if not (self.cfg.session_start_utc <= _hour < self.cfg.session_end_utc):
                    continue

            # ── 4. trend state (slow timeframe) — cached lookup ──────────
            # Find the latest trend bar whose time ≤ current signal bar time.
            _tidx = bisect.bisect_right(_trend_times, bar_time) - 1
            if _tidx < 0:
                continue
            _trend_t         = _trend_times[_tidx]
            _trend_df_row    = _trend_df_rows[_tidx]
            trend_state      = _trend_cache[_trend_t]
            last_trend_state = trend_state

            # update consecutive bear counter (count days, not bars)
            # Uses continuous direction threshold so BEAR_RANGING / DISTRIBUTION
            # also count — regime alone (BEAR_TRENDING) was insufficient.
            _today = bar_time.date() if hasattr(bar_time, "date") else bar_time
            if trend_state.direction < self.cfg.bear_filter_threshold:
                if _last_bear_date is None or _today != _last_bear_date:
                    consecutive_bear += 1
                    _last_bear_date = _today
            else:
                consecutive_bear = 0
                _last_bear_date  = None

            # ── 5. mid-timeframe direction (optional MTF filter) — cached ─
            mid_direction: Optional[float] = None
            if _mid_times:
                _midx = bisect.bisect_right(_mid_times, bar_time) - 1
                if _midx >= 0:
                    mid_direction = _mid_cache[_mid_times[_midx]]

            # ── 6. volatility regime ──────────────────────────────────────
            # Reconstruct trend_window using the cached df row index (iloc is a
            # zero-copy view; no boolean mask scan over full trend_df).
            trend_window = trend_df.iloc[max(0, _trend_df_row - 199): _trend_df_row + 1]
            vol_state  = None
            sig_window = sig_df.iloc[max(0, i - 200): i + 1]
            try:
                vol_state = vol_analyzer.analyse(trend_window, sig_window)
            except Exception:
                pass

            if vol_state is not None and vol_state.skip:
                continue

            # ── 7. signal detection ───────────────────────────────────────
            _is_high_vol = (
                vol_state is not None
                and str(vol_state.regime) in ("high", "crisis")
            )
            _min_conf = (
                self.cfg.min_confidence_high_vol if _is_high_vol
                else self.cfg.min_confidence
            )

            signals = signal_eng.analyse(
                sig_window,
                symbol=self.cfg.symbol,
                timeframe=self.cfg.timeframe,
                trend_state=trend_state,
                vol_state=vol_state,
                min_confidence=_min_conf,
            )
            if not signals:
                continue

            if vol_state is not None and vol_state.preferred_sources:
                signals = [
                    s for s in signals
                    if any(src in s.source for src in vol_state.preferred_sources)
                ]
            if not signals:
                continue

            best       = signals[0]
            entry_side = "buy" if best.signal_type == "long" else "sell"

            # ── 8a. bear market filter ───────────────────────────────────
            if self.cfg.bear_filter and consecutive_bear >= self.cfg.bear_filter_bars:
                if self.cfg.bear_filter_long_only and entry_side == "buy":
                    continue   # block longs in sustained bear trend
                elif not self.cfg.bear_filter_long_only:
                    continue   # block all new entries

            # ── 8b. cooldown filter ───────────────────────────────────────
            if sl_cooldown[entry_side] > 0:
                continue

            # ── 9. MTF confirmation (soft multiplier) ────────────────────
            # Hard-block only when mid-TF is strongly opposed (>0.30).
            # For the gray zone [-0.30, +0.30], scale confidence by alignment
            # strength: full agreement → 1.10×, neutral → 1.00×, mild opposition
            # → 0.85×.  This avoids discarding good signals on a weak MTF value.
            if mid_direction is not None:
                if best.signal_type == "long":
                    if mid_direction < -0.30:
                        continue        # strongly bearish 4h — hard block
                    mtf_scale = 1.0 + 0.10 * mid_direction   # range [0.97, 1.10]
                else:
                    if mid_direction > 0.30:
                        continue        # strongly bullish 4h — hard block
                    mtf_scale = 1.0 - 0.10 * mid_direction   # range [0.97, 1.10]
                mtf_scale = max(0.85, min(1.10, mtf_scale))
                scaled_conf = best.confidence * mtf_scale
                if scaled_conf < _min_conf:
                    continue
                best = dataclasses.replace(best, confidence=round(scaled_conf, 4))

            # ── 10. risk evaluation ───────────────────────────────────────
            portfolio = self._make_portfolio(capital, open_pos, close_price)
            decision  = risk_mgr.evaluate(best, portfolio, close_price, sig_window, trend_window)
            if not decision.approved or decision.position_size is None:
                continue

            pos        = decision.position_size
            _cur_vol_reg = str(vol_state.regime) if vol_state is not None else ""
            sl_cooldown_vol[entry_side] = _cur_vol_reg

            # ATR for trailing stop (computed from signal candles)
            _trail_atr = PositionSizer._atr(sig_window, close_price)

            # ── 11. entry: pullback limit or immediate fill ───────────────
            if self.cfg.pullback_entry:
                # Set limit order at a slight pullback from the close price
                pullback_dist = self.cfg.pullback_atr_frac * _trail_atr
                if entry_side == "buy":
                    limit_px = close_price - pullback_dist
                else:
                    limit_px = close_price + pullback_dist

                pending = _PendingEntry(
                    entry_side=entry_side,
                    limit_price=limit_px,
                    quantity=pos.quantity,
                    stop_price=pos.stop_price,
                    tp_price=pos.take_profit_price,
                    confidence=best.confidence,
                    regime=str(best.regime),
                    trail_atr=_trail_atr,
                    bars_left=self.cfg.pullback_max_bars,
                )
            else:
                # Immediate fill at close price + slippage
                if entry_side == "buy":
                    fill = close_price * (1 + SLIPPAGE_RATE)
                else:
                    fill = close_price * (1 - SLIPPAGE_RATE)
                slip     = abs(fill - close_price) * pos.quantity
                fee      = fill * pos.quantity * FEE_RATE
                capital -= fee

                open_pos = _OpenPos(
                    trade_id=uuid.uuid4(),
                    symbol=self.cfg.symbol,
                    side=entry_side,
                    entry_price=fill,
                    quantity=pos.quantity,
                    stop_price=pos.stop_price,
                    tp_price=pos.take_profit_price,
                    entry_time=bar_time,
                    entry_fee=fee,
                    entry_slip=slip,
                    confidence=best.confidence,
                    regime=str(best.regime),
                    trail_atr=_trail_atr,
                )

        # ── force-close any remaining open position at last bar ───────────
        if open_pos is not None and len(sig_df) > 0:
            last_bar  = sig_df.iloc[-1]
            last_time = last_bar.name if isinstance(last_bar.name, datetime) else last_bar["time"]
            ct = self._force_close(open_pos, float(last_bar["close"]), last_time)
            closed_trades.append(ct)

        # ── compute metrics ───────────────────────────────────────────────
        trade_results = [self._to_trade_result(t) for t in closed_trades]
        metrics = MetricsCalculator().calculate(trade_results, self.cfg.initial_capital)

        start_at = sig_df.iloc[MIN_CANDLES].name if len(sig_df) > MIN_CANDLES else None
        end_at   = sig_df.iloc[-1].name if len(sig_df) > 0 else None

        return BacktestResult(
            config=self.cfg,
            trades=closed_trades,
            metrics=metrics,
            start_at=start_at if isinstance(start_at, datetime) else None,
            end_at=end_at   if isinstance(end_at,   datetime) else None,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime index and required columns."""
        df = df.copy()
        if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _make_portfolio(capital: float, open_pos: Optional[_OpenPos], price: float) -> Portfolio:
        positions: dict[str, OpenPosition] = {}
        used = 0.0
        if open_pos is not None:
            op = OpenPosition(
                symbol=open_pos.symbol,
                side="long" if open_pos.side == "buy" else "short",
                quantity=open_pos.quantity,
                entry_price=open_pos.entry_price,
                current_price=price,
            )
            positions[open_pos.symbol] = op
            used = op.notional

        total = capital + sum(p.unrealized_pnl for p in positions.values())
        return Portfolio(
            total_value=total,
            cash=max(0.0, total - used),
            open_positions=positions,
            peak_value=max(capital, total),
            daily_pnl=0.0,
            daily_start_value=capital,
        )

    @staticmethod
    def _check_exit(
        pos:                    _OpenPos,
        bar:                    pd.Series,
        bar_time:               datetime,
        breakeven_trigger_pct:  float = 0.50,
        trailing_trigger_pct:   float = 0.75,
        trailing_atr_mult:      float = 1.5,
    ) -> Optional[BacktestTrade]:
        """Update stop management state and check for exit; return trade or None."""
        low   = float(bar["low"])
        high  = float(bar["high"])
        close = float(bar["close"])

        tp_dist = abs(pos.tp_price - pos.entry_price)

        # ── update best price ─────────────────────────────────────────────
        if pos.side == "buy":
            pos.best_price = max(pos.best_price, high)
            gained         = close - pos.entry_price
        else:
            pos.best_price = min(pos.best_price, low)
            gained         = pos.entry_price - close

        # ── breakeven activation ──────────────────────────────────────────
        if not pos.breakeven_activated and tp_dist > 0:
            if gained >= breakeven_trigger_pct * tp_dist:
                pos.active_stop         = pos.entry_price
                pos.breakeven_activated = True

        # ── trailing stop activation & update ────────────────────────────
        if pos.trail_atr > 0:
            trail_dist = trailing_atr_mult * pos.trail_atr

            if pos.breakeven_activated and not pos.trailing_active and tp_dist > 0:
                if gained >= trailing_trigger_pct * tp_dist:
                    pos.trailing_active = True

            if pos.trailing_active:
                if pos.side == "buy":
                    new_trail = pos.best_price - trail_dist
                    # trail only moves up (in favour), never down
                    pos.active_stop = max(pos.active_stop, new_trail)
                else:
                    new_trail = pos.best_price + trail_dist
                    # trail only moves down (in favour), never up
                    pos.active_stop = min(pos.active_stop, new_trail)

        # ── exit check ────────────────────────────────────────────────────
        exit_reason: Optional[str] = None
        exit_px:     float         = 0.0
        active_stop  = pos.active_stop

        if pos.side == "buy":
            if low <= active_stop:
                if pos.trailing_active:
                    exit_reason = "trailing_stop"
                elif pos.breakeven_activated:
                    exit_reason = "breakeven"
                else:
                    exit_reason = "stop_loss"
                exit_px = active_stop
            elif high >= pos.tp_price:
                exit_reason, exit_px = "take_profit", pos.tp_price
        else:
            if high >= active_stop:
                if pos.trailing_active:
                    exit_reason = "trailing_stop"
                elif pos.breakeven_activated:
                    exit_reason = "breakeven"
                else:
                    exit_reason = "stop_loss"
                exit_px = active_stop
            elif low <= pos.tp_price:
                exit_reason, exit_px = "take_profit", pos.tp_price

        if exit_reason is None:
            return None

        return BacktestEngine._build_trade(pos, exit_px, bar_time, exit_reason)

    @staticmethod
    def _force_close(pos: _OpenPos, price: float, t: datetime) -> BacktestTrade:
        return BacktestEngine._build_trade(pos, price, t, "end_of_data")

    @staticmethod
    def _build_trade(
        pos: _OpenPos, exit_px: float, exit_time: datetime, reason: str,
        qty_override: Optional[float] = None,
    ) -> BacktestTrade:
        qty = qty_override if qty_override is not None else pos.quantity

        if pos.side == "buy":
            fill = exit_px * (1 - SLIPPAGE_RATE)
        else:
            fill = exit_px * (1 + SLIPPAGE_RATE)

        exit_slip  = abs(fill - exit_px) * qty
        exit_fee   = fill * qty * FEE_RATE
        # entry fee is proportional when closing a partial quantity
        entry_fee_share = pos.entry_fee * (qty / (pos.quantity + qty)) if qty_override else pos.entry_fee
        total_fee  = entry_fee_share + exit_fee
        total_slip = (pos.entry_slip * (qty / (pos.quantity + qty)) if qty_override else pos.entry_slip) + exit_slip

        if pos.side == "buy":
            pnl = (fill - pos.entry_price) * qty - total_fee
        else:
            pnl = (pos.entry_price - fill) * qty - total_fee

        notional = pos.entry_price * qty
        pnl_pct  = pnl / notional if notional > 0 else 0.0

        return BacktestTrade(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=fill,
            quantity=qty,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            fees=round(total_fee, 6),
            slippage=round(total_slip, 6),
            exit_reason=reason,
            stop_price=pos.stop_price,
            tp_price=pos.tp_price,
            confidence=pos.confidence,
            regime=pos.regime,
        )

    @staticmethod
    def _check_partial(
        pos:       _OpenPos,
        bar:       pd.Series,
        bar_time:  datetime,
        r_mult:    float,
        close_pct: float,
    ) -> Optional[BacktestTrade]:
        """Check if partial profit target (N×R) is hit; if so, close `close_pct` of
        the position and immediately lock breakeven on the remainder.  Mutates pos."""
        sl_dist = abs(pos.entry_price - pos.stop_price)
        if sl_dist == 0:
            return None

        if pos.side == "buy":
            target = pos.entry_price + r_mult * sl_dist
            hit    = float(bar["high"]) >= target
        else:
            target = pos.entry_price - r_mult * sl_dist
            hit    = float(bar["low"])  <= target

        if not hit:
            return None

        partial_qty      = pos.quantity * close_pct
        pos.quantity    -= partial_qty          # shrink remaining position
        pos.partial_closed    = True
        # lock breakeven on the remainder immediately
        pos.active_stop       = pos.entry_price
        pos.breakeven_activated = True

        return BacktestEngine._build_trade(
            pos, target, bar_time, "partial_tp", qty_override=partial_qty
        )

    @staticmethod
    def _to_trade_result(bt: BacktestTrade) -> TradeResult:
        return TradeResult(
            pnl=bt.pnl,
            pnl_pct=bt.pnl_pct,
            opened_at=bt.entry_time,
            closed_at=bt.exit_time,
            fees=bt.fees,
            slippage=bt.slippage,
            side=bt.side,
        )
