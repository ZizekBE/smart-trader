"""smart-trader entry point.

Registered as the `trader` CLI command in pyproject.toml.
Reads all configuration from environment variables (see Settings).

Usage:
    uv run trader                         # paper trading, BTC/USDT, 1h/4h/1d
    uv run trader --symbol ETH/USDT       # different symbol
    TRADING_MODE=live uv run trader       # live mode (use with caution)
"""
from __future__ import annotations

import argparse
import asyncio
import signal

import structlog

from smart_trader.core.settings import get_settings
from smart_trader.trader.loop import TradingLoop
from smart_trader.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="smart-trader — AI-powered trading engine")
    p.add_argument("--symbol",    default=None, help="Trading pair, e.g. BTC/USDT")
    p.add_argument("--symbols",   nargs="+", default=None,
                   help="Multiple symbols for multi mode, e.g. ETH/USDT BTC/USDT")
    p.add_argument("--signal-tf", default="1h", help="Signal timeframe (default: 1h)")
    p.add_argument("--trend-tf",  default="1d", help="Trend timeframe  (default: 1d)")
    p.add_argument("--mid-tf",    default="4h", help="Mid timeframe    (default: 4h)")
    p.add_argument("--cash",      type=float, default=None, help="Initial virtual cash (total across all symbols)")
    p.add_argument("--once",      action="store_true", help="Run a single tick and exit")
    p.add_argument("--log-level", default=None, choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--mode",      default="rule", choices=["rule", "rl", "shadow", "hybrid", "multi"],
                   help="Trading mode: rule-based, RL, shadow, hybrid dual-sleeve, or multi-symbol hybrid")
    p.add_argument("--model-path", default=None, help="Path to trained RL agent checkpoint")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    s = get_settings()

    log_level = args.log_level or s.log_level
    configure_logging(log_level)

    symbol = args.symbol or (s.symbols[0] if s.symbols else "BTC/USDT")
    cash   = args.cash   or 10_000.0
    paper  = s.trading_mode != "live"

    log.info(
        "smart_trader_starting",
        symbol=symbol, signal_tf=args.signal_tf,
        trend_tf=args.trend_tf, mid_tf=args.mid_tf,
        mode="paper" if paper else "LIVE",
        engine=args.mode,
    )

    if args.mode in ("rl", "shadow"):
        await _run_rl(args, symbol, cash, paper)
    elif args.mode == "hybrid":
        await _run_hybrid(args, symbol, cash, paper)
    elif args.mode == "multi":
        symbols = args.symbols or ["ETH/USDT", "BTC/USDT"]
        await _run_multi(args, symbols, cash, paper)
    else:
        await _run_rule(args, symbol, cash, paper)


async def _run_rule(args, symbol: str, cash: float, paper: bool) -> None:
    loop = TradingLoop(
        symbol=symbol,
        signal_tf=args.signal_tf,
        trend_tf=args.trend_tf,
        mid_tf=args.mid_tf,
        initial_cash=cash,
        paper=paper,
    )

    if args.once:
        await loop.run_once()
        return

    # graceful shutdown on SIGINT / SIGTERM
    stop = asyncio.Event()
    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, stop.set)

    run_task = asyncio.create_task(loop.run())
    await stop.wait()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    log.info("shutdown_complete")


async def _run_rl(args, symbol: str, cash: float, paper: bool) -> None:
    from smart_trader.agent.inference import InferenceConfig
    from smart_trader.data.features.store import FeatureStore
    from smart_trader.exchange.factory import create_adapter
    from smart_trader.trader.rl_loop import RLTradingLoop

    adapter = create_adapter(paper=paper)
    feature_store = FeatureStore()
    inference_config = InferenceConfig(
        model_path=args.model_path or "",
        shadow_mode=(args.mode == "shadow"),
    )

    rl_loop = RLTradingLoop(
        adapter=adapter,
        feature_store=feature_store,
        inference_config=inference_config,
        symbols=[symbol],
        initial_capital=cash,
    )

    stop = asyncio.Event()
    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, stop.set)

    run_task = asyncio.create_task(rl_loop.start())
    await stop.wait()
    await rl_loop.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    log.info("shutdown_complete")


async def _run_hybrid(args, symbol: str, cash: float, paper: bool) -> None:
    from smart_trader.trader.hybrid_loop import HybridLoop

    loop = HybridLoop(symbol=symbol, initial_cash=cash, paper=True)

    stop = asyncio.Event()
    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, stop.set)

    if args.once:
        await loop.run_once()
        return

    run_task = asyncio.create_task(loop.run())
    await stop.wait()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    log.info("shutdown_complete")


async def _run_multi(args, symbols: list[str], cash: float, paper: bool) -> None:
    from smart_trader.trader.multi_loop import MultiHybridLoop

    loop = MultiHybridLoop(symbols=symbols, total_capital=cash, paper=paper)

    log.info("smart_trader_starting", symbols=symbols, mode="multi", paper=paper,
             capital_per_symbol=cash / len(symbols))

    stop = asyncio.Event()
    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, stop.set)

    if args.once:
        await loop.run_once()
        return

    run_task = asyncio.create_task(loop.run())
    await stop.wait()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    log.info("shutdown_complete")


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
