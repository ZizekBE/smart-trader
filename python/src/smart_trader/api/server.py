"""FastAPI server — exposes trading engine data and control to the UI.

Endpoints
─────────
GET  /api/candles              — OHLCV candles from DB
POST /api/backtest             — run backtest, auto-save report, return result
GET  /api/reports              — list saved backtest reports (summary)
GET  /api/reports/{id}         — load a full saved report for replay
DELETE /api/reports/{id}       — delete a report
GET  /api/trades               — closed trades from DB
GET  /api/positions            — open positions from DB
GET  /api/trader/status        — is TradingLoop running?
POST /api/trader/start         — start TradingLoop (paper mode)
POST /api/trader/stop          — stop TradingLoop

Run:
    uvicorn smart_trader.api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from smart_trader.analysis.backtest.engine import BacktestConfig, BacktestEngine
from smart_trader.api.state import signal_log
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory
from smart_trader.data.storage.trade_repo import TradeRepository

log = structlog.get_logger(__name__)

app = FastAPI(title="smart-trader", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_factory = get_session_factory()

# ── trader lifecycle ─────────────────────────────────────────────────────────

_trader_task:   Optional[asyncio.Task] = None
_trader_symbol: Optional[str]          = None
_trader_mode:   str                    = "single"


# ── request / response models ────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol:           str   = "BTC/USDT"
    timeframe:        str   = "1h"
    trend_timeframe:  str   = "1d"
    mid_timeframe:    str   = "4h"
    strategy_version: str   = "v2"
    initial_capital:  float = 10_000.0
    min_confidence:   float = 0.65
    days:             int   = 180


class HybridBacktestRequest(BaseModel):
    symbol:           str        = "BTC/USDT"
    symbols:          list[str]  = []          # if non-empty, run multi-symbol
    strategy_version: str        = "v2"
    initial_capital:  float      = 10_000.0
    days:             int        = 180


class TraderStartRequest(BaseModel):
    symbol:    str   = "BTC/USDT"
    signal_tf: str   = "1h"
    trend_tf:  str   = "1d"
    mid_tf:    str   = "4h"
    cash:      float = 10_000.0
    mode:      str   = "single"   # "single" | "hybrid"


# ── helpers ───────────────────────────────────────────────────────────────────

def _candle_to_dict(c) -> dict:
    return {
        "time":   int(c.time.timestamp()),
        "open":   float(c.open),
        "high":   float(c.high),
        "low":    float(c.low),
        "close":  float(c.close),
        "volume": float(c.volume),
    }


# ── candles ───────────────────────────────────────────────────────────────────

@app.get("/api/candles")
async def get_candles(
    symbol:    str           = Query("BTC/USDT"),
    tf:        str           = Query("1h"),
    limit:     int           = Query(1000, ge=10, le=5000),
    exchange:  str           = Query("gateio"),
    since:     Optional[str] = Query(None),   # ISO-8601 date, e.g. "2025-10-01"
):
    now = datetime.now(timezone.utc)
    if since:
        since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    else:
        days     = {"1m": 2, "5m": 7, "15m": 14, "1h": 180, "4h": 540, "1d": 730}.get(tf, 180)
        since_dt = now - timedelta(days=days)

    async with _factory() as session:
        repo    = CandleRepository(session)
        candles = await repo.get_range(symbol, exchange, tf, since_dt, now)

    return [_candle_to_dict(c) for c in candles[-limit:]]


# ── backtest ──────────────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    now   = datetime.now(timezone.utc)
    since = now - timedelta(days=req.days)

    async with _factory() as session:
        repo          = CandleRepository(session)
        sig_candles   = await repo.get_range(req.symbol, "gateio", req.timeframe, since, now)
        trend_candles = await repo.get_range(req.symbol, "gateio", req.trend_timeframe, since, now)
        mid_candles   = await repo.get_range(req.symbol, "gateio", req.mid_timeframe, since, now)

    if not sig_candles:
        raise HTTPException(404, f"No candles for {req.symbol} {req.timeframe}")

    import pandas as pd

    def _to_df(candles):
        if not candles:
            return None
        rows = [{"time": c.time, "open": float(c.open), "high": float(c.high),
                 "low": float(c.low), "close": float(c.close), "volume": float(c.volume)}
                for c in candles]
        df = pd.DataFrame(rows).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    sig_df   = _to_df(sig_candles)
    trend_df = _to_df(trend_candles)
    mid_df   = _to_df(mid_candles)

    cfg = BacktestConfig(
        symbol=req.symbol,
        timeframe=req.timeframe,
        trend_timeframe=req.trend_timeframe,
        initial_capital=req.initial_capital,
        min_confidence=req.min_confidence,
        strategy_version=req.strategy_version,
    )

    # run CPU-bound work in thread pool to avoid blocking the event loop
    result = await asyncio.to_thread(BacktestEngine(cfg).run, sig_df, trend_df, mid_df)

    m = result.metrics
    trades = [
        {
            "id":          str(t.trade_id)[:8],
            "side":        t.side,
            "entry_time":  int(t.entry_time.timestamp()),
            "exit_time":   int(t.exit_time.timestamp()),
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "quantity":    t.quantity,
            "pnl":         round(t.pnl, 4),
            "pnl_pct":     round(t.pnl_pct * 100, 3),
            "exit_reason": t.exit_reason,
            "confidence":  t.confidence,
            "regime":      t.regime,
            "stop_price":  t.stop_price,
            "tp_price":    t.tp_price,
        }
        for t in result.trades
    ]

    metrics_dict = {
        "total_trades":     m.total_trades,
        "win_rate":         round(m.win_rate * 100, 1),
        "total_pnl":        round(m.total_pnl, 2),
        "total_pnl_pct":    round(m.total_pnl_pct * 100, 2),
        "sharpe_ratio":     round(m.sharpe_ratio, 2),
        "sortino_ratio":    round(m.sortino_ratio, 2),
        "max_drawdown_pct": round(m.max_drawdown_pct * 100, 2),
        "profit_factor":    round(m.profit_factor, 2) if m.profit_factor != float("inf") else 999,
        "expectancy":       round(m.expectancy, 4),
        "avg_win":            round(m.avg_win, 4),
        "avg_loss":           round(m.avg_loss, 4),
        "annualized_return":  round(m.annualized_return * 100, 1),
    }

    # ── auto-save report ──────────────────────────────────────────────────
    report_id = str(uuid.uuid4())
    config_snapshot = {
        "symbol": req.symbol, "timeframe": req.timeframe,
        "strategy_version": req.strategy_version, "days": req.days,
        "initial_capital": req.initial_capital, "min_confidence": req.min_confidence,
    }
    try:
        async with _factory() as session:
            await session.execute(
                __import__("sqlalchemy", fromlist=["text"]).text("""
                    INSERT INTO backtest_reports
                        (id, symbol, timeframe, strategy_version, days,
                         start_at, end_at, metrics, trades, config)
                    VALUES
                        (:id, :symbol, :timeframe, :strategy_version, :days,
                         :start_at, :end_at, :metrics, :trades, :config)
                """),
                {
                    "id":               report_id,
                    "symbol":           req.symbol,
                    "timeframe":        req.timeframe,
                    "strategy_version": req.strategy_version,
                    "days":             req.days,
                    "start_at":         result.start_at,
                    "end_at":           result.end_at,
                    "metrics":          json.dumps(metrics_dict),
                    "trades":           json.dumps(trades),
                    "config":           json.dumps(config_snapshot),
                },
            )
            await session.commit()
        log.info("report_saved", id=report_id, symbol=req.symbol)
    except Exception as exc:
        log.warning("report_save_failed", error=str(exc))
        report_id = None

    return {
        "report_id": report_id,
        "trades":    trades,
        "metrics":   metrics_dict,
        "start_at":  result.start_at.isoformat() if result.start_at else None,
        "end_at":    result.end_at.isoformat()   if result.end_at   else None,
    }


# ── hybrid backtest ───────────────────────────────────────────────────────────

@app.post("/api/backtest/hybrid")
async def run_hybrid_backtest(req: HybridBacktestRequest):
    from smart_trader.analysis.backtest.hybrid_engine import HybridBacktestEngine
    from smart_trader.analysis.metrics.calculator import MetricsCalculator, TradeResult
    from sqlalchemy import text as sa_text

    import pandas as pd

    symbols = req.symbols if req.symbols else [req.symbol]
    now     = datetime.now(timezone.utc)
    since   = now - timedelta(days=req.days)

    def _to_df(candles):
        if not candles:
            return None
        rows = [{"time": c.time, "open": float(c.open), "high": float(c.high),
                 "low": float(c.low), "close": float(c.close), "volume": float(c.volume)}
                for c in candles]
        df = pd.DataFrame(rows).set_index("time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def _load_core_params(sym: str) -> dict | None:
        async with _factory() as session:
            row = (await session.execute(
                sa_text("SELECT best_params FROM optimization_runs WHERE symbol=:s AND timeframe='4h_core' AND is_current=TRUE ORDER BY id DESC LIMIT 1"),
                {"s": sym},
            )).fetchone()
            return row.best_params if row else None

    def _metrics_dict(m):
        return {
            "total_trades":     m.total_trades,
            "win_rate":         round(m.win_rate * 100, 2),
            "total_pnl":        round(m.total_pnl, 2),
            "total_pnl_pct":    round(m.total_pnl_pct * 100, 2),
            "sharpe_ratio":     round(m.sharpe_ratio, 2),
            "sortino_ratio":    round(m.sortino_ratio, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct * 100, 2),
            "profit_factor":    round(m.profit_factor, 2) if m.profit_factor != float("inf") else 999,
            "expectancy":       round(m.expectancy, 4),
            "avg_win":            round(m.avg_win, 4),
            "avg_loss":           round(m.avg_loss, 4),
            "annualized_return":  round(m.annualized_return * 100, 1),
        }

    def _trade_list(trades):
        return [
            {
                "id":          str(t.trade_id)[:8],
                "side":        t.side,
                "entry_time":  int(t.entry_time.timestamp()),
                "exit_time":   int(t.exit_time.timestamp()),
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "pnl":         round(t.pnl, 4),
                "pnl_pct":     round(t.pnl_pct * 100, 3),
                "exit_reason": t.exit_reason,
                "confidence":  t.confidence,
                "regime":      t.regime,
                "stop_price":  t.stop_price,
                "tp_price":    t.tp_price,
            }
            for t in trades
        ]

    # ── run per symbol ────────────────────────────────────────────────────
    sym_results: dict = {}
    all_trades_flat   = []

    for sym in symbols:
        async with _factory() as session:
            repo = CandleRepository(session)
            h1_candles = await repo.get_range(sym, "gateio", "1h", since, now)
            h4_candles = await repo.get_range(sym, "gateio", "4h", since, now)
            d1_candles = await repo.get_range(sym, "gateio", "1d", since, now)

        if not h1_candles or not h4_candles or not d1_candles:
            log.warning("hybrid_backtest_no_candles", symbol=sym)
            continue

        h1_df = _to_df(h1_candles)
        h4_df = _to_df(h4_candles)
        d1_df = _to_df(d1_candles)
        core_params = await _load_core_params(sym)

        engine = HybridBacktestEngine(
            symbol=req.symbol if len(symbols) == 1 else sym,
            initial_capital=req.initial_capital,
            strategy_version=req.strategy_version,
            core_params=core_params,
        )
        result = await asyncio.to_thread(engine.run, h1_df, h4_df, d1_df)

        sym_results[sym] = {
            "core": {
                "trades":  _trade_list(result.core.trades),
                "metrics": _metrics_dict(result.core.metrics),
                "config": {
                    "timeframe":       result.core.config.timeframe,
                    "min_confidence":  result.core.config.min_confidence,
                    "atr_mult":        result.core.config.atr_mult,
                    "rr_ratio":        result.core.config.rr_ratio,
                    "initial_capital": result.core.config.initial_capital,
                },
            },
            "tactical": {
                "trades":  _trade_list(result.tactical.trades),
                "metrics": _metrics_dict(result.tactical.metrics),
                "config": {
                    "timeframe":       result.tactical.config.timeframe,
                    "min_confidence":  result.tactical.config.min_confidence,
                    "atr_mult":        result.tactical.config.atr_mult,
                    "rr_ratio":        result.tactical.config.rr_ratio,
                    "initial_capital": result.tactical.config.initial_capital,
                },
            },
            "combined": _metrics_dict(result.combined),
            "trades":   _trade_list(result.all_trades),
        }
        all_trades_flat.extend(result.all_trades)

    if not sym_results:
        raise HTTPException(404, "No candle data available for any symbol")

    # ── aggregate across all symbols ──────────────────────────────────────
    all_trades_flat.sort(key=lambda t: t.entry_time)
    agg_trade_results = [
        TradeResult(pnl=t.pnl, pnl_pct=t.pnl_pct, opened_at=t.entry_time,
                    closed_at=t.exit_time, fees=t.fees, slippage=t.slippage, side=t.side)
        for t in all_trades_flat
    ]
    total_capital   = req.initial_capital * len(sym_results)
    aggregate       = MetricsCalculator().calculate(agg_trade_results, total_capital)
    all_trades_ser  = _trade_list(all_trades_flat)

    # ── for backward-compat single-symbol response ────────────────────────
    primary = sym_results[symbols[0]] if symbols[0] in sym_results else next(iter(sym_results.values()))
    first_result_start = None
    first_result_end   = None

    # ── auto-save report ──────────────────────────────────────────────────
    report_id = str(uuid.uuid4())
    agg_metrics = _metrics_dict(aggregate)
    try:
        async with _factory() as session:
            await session.execute(sa_text("""
                INSERT INTO backtest_reports
                    (id, symbol, timeframe, strategy_version, days,
                     start_at, end_at, metrics, trades, config)
                VALUES
                    (:id, :symbol, :timeframe, :strategy_version, :days,
                     :start_at, :end_at, :metrics, :trades, :config)
            """), {
                "id":               report_id,
                "symbol":           ", ".join(symbols),
                "timeframe":        "hybrid",
                "strategy_version": req.strategy_version,
                "days":             req.days,
                "start_at":         None,
                "end_at":           None,
                "metrics":          json.dumps(agg_metrics),
                "trades":           json.dumps(all_trades_ser),
                "config":           json.dumps({"symbols": symbols, "days": req.days,
                                                "mode": "hybrid", "initial_capital": req.initial_capital}),
            })
            await session.commit()
    except Exception as exc:
        log.warning("hybrid_report_save_failed", error=str(exc))
        report_id = None

    return {
        "report_id": report_id,
        # single-symbol compat (first / only symbol)
        "core":     primary["core"],
        "tactical": primary["tactical"],
        "combined": primary["combined"] if len(sym_results) == 1 else agg_metrics,
        "trades":   all_trades_ser,
        "start_at": first_result_start,
        "end_at":   first_result_end,
        # multi-symbol extras
        "symbols":   sym_results,
        "aggregate": agg_metrics if len(sym_results) > 1 else None,
    }

    def _metrics_dict(m):
        return {
            "total_trades":     m.total_trades,
            "win_rate":         round(m.win_rate * 100, 2),
            "total_pnl":        round(m.total_pnl, 2),
            "total_pnl_pct":    round(m.total_pnl_pct * 100, 2),
            "sharpe_ratio":     round(m.sharpe_ratio, 2),
            "sortino_ratio":    round(m.sortino_ratio, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct * 100, 2),
            "profit_factor":    round(m.profit_factor, 2) if m.profit_factor != float("inf") else 999,
            "expectancy":       round(m.expectancy, 4),
            "avg_win":            round(m.avg_win, 4),
            "avg_loss":           round(m.avg_loss, 4),
            "annualized_return":  round(m.annualized_return * 100, 1),
        }

    def _trade_list(trades):
        return [
            {
                "id":          str(t.trade_id)[:8],
                "side":        t.side,
                "entry_time":  int(t.entry_time.timestamp()),
                "exit_time":   int(t.exit_time.timestamp()),
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "pnl":         round(t.pnl, 4),
                "pnl_pct":     round(t.pnl_pct * 100, 3),
                "exit_reason": t.exit_reason,
                "confidence":  t.confidence,
                "regime":      t.regime,
                "stop_price":  t.stop_price,
                "tp_price":    t.tp_price,
            }
            for t in trades
        ]

    all_trades_serialized  = _trade_list(result.all_trades)
    combined_metrics = _metrics_dict(result.combined)

    # auto-save as a report (combined view)
    report_id = str(uuid.uuid4())
    try:
        async with _factory() as session:
            await session.execute(sa_text("""
                INSERT INTO backtest_reports
                    (id, symbol, timeframe, strategy_version, days,
                     start_at, end_at, metrics, trades, config)
                VALUES
                    (:id, :symbol, :timeframe, :strategy_version, :days,
                     :start_at, :end_at, :metrics, :trades, :config)
            """), {
                "id":               report_id,
                "symbol":           req.symbol,
                "timeframe":        "hybrid",
                "strategy_version": req.strategy_version,
                "days":             req.days,
                "start_at":         result.start_at,
                "end_at":           result.end_at,
                "metrics":          json.dumps(combined_metrics),
                "trades":           json.dumps(all_trades_serialized),
                "config":           json.dumps({"symbol": req.symbol, "days": req.days,
                                                "mode": "hybrid", "initial_capital": req.initial_capital}),
            })
            await session.commit()
    except Exception as exc:
        log.warning("hybrid_report_save_failed", error=str(exc))
        report_id = None

    return {
        "report_id": report_id,
        "core": {
            "trades":  _trade_list(result.core.trades),
            "metrics": _metrics_dict(result.core.metrics),
            "config": {
                "timeframe":    result.core.config.timeframe,
                "min_confidence": result.core.config.min_confidence,
                "atr_mult":     result.core.config.atr_mult,
                "rr_ratio":     result.core.config.rr_ratio,
                "initial_capital": result.core.config.initial_capital,
            },
        },
        "tactical": {
            "trades":  _trade_list(result.tactical.trades),
            "metrics": _metrics_dict(result.tactical.metrics),
            "config": {
                "timeframe":    result.tactical.config.timeframe,
                "min_confidence": result.tactical.config.min_confidence,
                "atr_mult":     result.tactical.config.atr_mult,
                "rr_ratio":     result.tactical.config.rr_ratio,
                "initial_capital": result.tactical.config.initial_capital,
            },
        },
        "combined": combined_metrics,
        "trades":   all_trades_serialized,
        "start_at": result.start_at.isoformat() if result.start_at else None,
        "end_at":   result.end_at.isoformat()   if result.end_at   else None,
    }


# ── backtest reports ──────────────────────────────────────────────────────────

@app.get("/api/reports")
async def list_reports(
    symbol: Optional[str] = None,
    limit:  int           = Query(50, ge=1, le=200),
):
    """Return summary list of saved backtest reports (no trade details)."""
    from sqlalchemy import text
    where = "WHERE symbol = :symbol" if symbol else ""
    async with _factory() as session:
        rows = await session.execute(
            text(f"""
                SELECT id, created_at, name, symbol, timeframe,
                       strategy_version, days, start_at, end_at, metrics
                FROM backtest_reports
                {where}
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"symbol": symbol, "limit": limit} if symbol else {"limit": limit},
        )
        results = rows.mappings().all()

    return [
        {
            "id":               str(r["id"]),
            "created_at":       r["created_at"].isoformat(),
            "name":             r["name"],
            "symbol":           r["symbol"],
            "timeframe":        r["timeframe"],
            "strategy_version": r["strategy_version"],
            "days":             r["days"],
            "start_at":         r["start_at"].isoformat() if r["start_at"] else None,
            "end_at":           r["end_at"].isoformat()   if r["end_at"]   else None,
            "metrics":          r["metrics"],
        }
        for r in results
    ]


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Load a full saved report including all trades (for chart replay)."""
    from sqlalchemy import text
    async with _factory() as session:
        row = (await session.execute(
            text("SELECT * FROM backtest_reports WHERE id = :id"),
            {"id": report_id},
        )).mappings().first()

    if row is None:
        raise HTTPException(404, f"Report {report_id} not found")

    return {
        "id":               str(row["id"]),
        "created_at":       row["created_at"].isoformat(),
        "name":             row["name"],
        "symbol":           row["symbol"],
        "timeframe":        row["timeframe"],
        "strategy_version": row["strategy_version"],
        "days":             row["days"],
        "start_at":         row["start_at"].isoformat() if row["start_at"] else None,
        "end_at":           row["end_at"].isoformat()   if row["end_at"]   else None,
        "metrics":          row["metrics"],
        "trades":           row["trades"],
        "config":           row["config"],
    }


@app.patch("/api/reports/{report_id}")
async def rename_report(report_id: str, body: dict):
    """Set a human-readable name on a saved report."""
    from sqlalchemy import text
    name = body.get("name", "").strip()[:120]
    async with _factory() as session:
        result = await session.execute(
            text("UPDATE backtest_reports SET name = :name WHERE id = :id"),
            {"name": name or None, "id": report_id},
        )
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, f"Report {report_id} not found")
    return {"id": report_id, "name": name}


@app.delete("/api/reports/{report_id}", status_code=204)
async def delete_report(report_id: str):
    from sqlalchemy import text
    async with _factory() as session:
        await session.execute(
            text("DELETE FROM backtest_reports WHERE id = :id"),
            {"id": report_id},
        )
        await session.commit()


# ── live trades ───────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def get_trades(
    symbol: Optional[str] = None,
    limit:  int           = Query(50, ge=1, le=500),
):
    async with _factory() as session:
        repo   = TradeRepository(session)
        trades = await repo.get_closed(symbol=symbol, limit=limit)

    return [
        {
            "id":          str(t.id)[:8],
            "symbol":      t.symbol,
            "side":        t.side,
            "entry_price": float(t.entry_price or 0),
            "exit_price":  float(t.exit_price or 0),
            "quantity":    float(t.quantity or 0),
            "pnl":         float(t.pnl or 0),
            "pnl_pct":     float(t.pnl_pct or 0) * 100,
            "exit_reason": t.exit_reason,
            "opened_at":   int(t.opened_at.timestamp()) if t.opened_at else None,
            "closed_at":   int(t.closed_at.timestamp()) if t.closed_at else None,
        }
        for t in trades
    ]


# ── positions ─────────────────────────────────────────────────────────────────

@app.get("/api/positions")
async def get_positions(symbol: Optional[str] = None):
    async with _factory() as session:
        repo   = TradeRepository(session)
        trades = await repo.get_open(symbol=symbol)

    return [
        {
            "id":          str(t.id)[:8],
            "symbol":      t.symbol,
            "side":        t.side,
            "entry_price": float(t.entry_price or 0),
            "quantity":    float(t.quantity or 0),
            "stop_price":  float(t.stop_price or 0),
            "tp_price":    float(t.take_profit_price or 0),
            "opened_at":   int(t.opened_at.timestamp()) if t.opened_at else None,
            "confidence":  float(t.confidence or 0),
            "sleeve":      getattr(t, "sleeve", "tactical"),
        }
        for t in trades
    ]


# ── trader control ────────────────────────────────────────────────────────────

@app.get("/api/trader/status")
async def trader_status():
    running = _trader_task is not None and not _trader_task.done()
    return {
        "running": running,
        "symbol":  _trader_symbol if running else None,
        "mode":    _trader_mode   if running else None,
    }


@app.post("/api/trader/start")
async def start_trader(req: TraderStartRequest):
    global _trader_task, _trader_symbol

    if _trader_task is not None and not _trader_task.done():
        raise HTTPException(409, "Trader already running")

    if req.mode == "hybrid":
        from smart_trader.trader.hybrid_loop import HybridLoop
        loop = HybridLoop(
            symbol=req.symbol,
            signal_tf=req.signal_tf,
            initial_cash=req.cash,
            paper=True,
            signal_callback=lambda rec: signal_log.appendleft(rec),
        )
    else:
        from smart_trader.trader.loop import TradingLoop
        loop = TradingLoop(
            symbol=req.symbol,
            signal_tf=req.signal_tf,
            trend_tf=req.trend_tf,
            mid_tf=req.mid_tf,
            initial_cash=req.cash,
            paper=True,
            signal_callback=lambda rec: signal_log.appendleft(rec),
        )
    _trader_task   = asyncio.create_task(loop.run())
    _trader_symbol = req.symbol
    _trader_mode   = req.mode
    log.info("trader_started", symbol=req.symbol, mode=req.mode)
    return {"status": "started", "symbol": req.symbol, "mode": req.mode}


@app.post("/api/trader/stop")
async def stop_trader():
    global _trader_task, _trader_symbol

    if _trader_task is None or _trader_task.done():
        raise HTTPException(409, "Trader is not running")

    _trader_task.cancel()
    try:
        await _trader_task
    except asyncio.CancelledError:
        pass

    _trader_task   = None
    _trader_symbol = None
    log.info("trader_stopped")
    return {"status": "stopped"}


@app.get("/api/optimization/latest")
async def get_optimization_latest(
    symbol:    str = Query("BTC/USDT"),
    timeframe: str = Query("1h"),
):
    """Return the most recent optimization run for a symbol/timeframe."""
    async with _factory() as session:
        rows = await session.execute(text("""
            SELECT id, created_at, study_name, train_start, train_end, test_start, test_end,
                   n_trials, best_params, train_score, test_score,
                   train_sharpe, test_sharpe, train_pnl, test_pnl, train_trades, test_trades
            FROM optimization_runs
            WHERE symbol = :s AND timeframe = :tf AND is_current = TRUE
            ORDER BY created_at DESC
            LIMIT 1
        """), {"s": symbol, "tf": timeframe})
        row = rows.fetchone()
    if row is None:
        return None
    keys = ["id", "created_at", "study_name", "train_start", "train_end",
            "test_start", "test_end", "n_trials", "best_params",
            "train_score", "test_score", "train_sharpe", "test_sharpe",
            "train_pnl", "test_pnl", "train_trades", "test_trades"]
    d = dict(zip(keys, row))
    d["id"] = str(d["id"])
    for k in ("created_at", "train_start", "train_end", "test_start", "test_end"):
        if d[k] is not None:
            d[k] = d[k].isoformat()
    return d


@app.get("/api/trader/sleeve-stats")
async def get_sleeve_stats(symbol: Optional[str] = None):
    """Per-sleeve open positions count + cumulative closed PnL for the live trader."""
    from sqlalchemy import text

    sym_filter = "AND symbol = :symbol" if symbol else ""
    params: dict = {"symbol": symbol} if symbol else {}

    async with _factory() as session:
        open_rows = (await session.execute(
            text(f"SELECT sleeve, COUNT(*) AS cnt FROM trades WHERE status='open' {sym_filter} GROUP BY sleeve"),
            params,
        )).fetchall()
        closed_rows = (await session.execute(
            text(f"SELECT sleeve, COUNT(*) AS trades, COALESCE(SUM(pnl),0) AS pnl FROM trades WHERE status='closed' {sym_filter} GROUP BY sleeve"),
            params,
        )).fetchall()

    open_map   = {r.sleeve: int(r.cnt)          for r in open_rows}
    closed_map = {r.sleeve: (int(r.trades), float(r.pnl)) for r in closed_rows}

    return [
        {
            "sleeve":         sleeve,
            "open_positions": open_map.get(sleeve, 0),
            "closed_trades":  closed_map.get(sleeve, (0, 0.0))[0],
            "pnl":            round(closed_map.get(sleeve, (0, 0.0))[1], 2),
        }
        for sleeve in ("core", "tactical")
    ]


@app.get("/api/trader/signals")
async def get_trader_signals(limit: int = Query(50, ge=1, le=100)):
    """Return recent per-tick signal analysis records (newest first)."""
    return list(signal_log)[:limit]


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("smart_trader.api.server:app", host="0.0.0.0", port=8000, reload=True)
