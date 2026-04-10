"""
smart-trader — paper trading entry point.

Usage:
    # paper mode (default), run continuously
    python scripts/run_trader.py

    # run a single tick immediately (useful for testing)
    python scripts/run_trader.py --once

    # change symbol / timeframe
    python scripts/run_trader.py --symbol ETH/USDT --signal-tf 1h
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smart_trader.core.env_files import load_repo_env_files

load_repo_env_files()

from smart_trader.trader.loop import TradingLoop
from smart_trader.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="smart-trader paper loop")
    p.add_argument("--symbol",    default="BTC/USDT")
    p.add_argument("--signal-tf", default="1h")
    p.add_argument("--trend-tf",  default="1d")
    p.add_argument("--mid-tf",    default="4h")
    p.add_argument("--cash",      type=float, default=10_000.0)
    p.add_argument("--once",      action="store_true",
                   help="run one tick then exit (no sleep)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    trader = TradingLoop(
        symbol=args.symbol,
        signal_tf=args.signal_tf,
        trend_tf=args.trend_tf,
        mid_tf=args.mid_tf,
        initial_cash=args.cash,
        paper=True,
    )

    if args.once:
        await trader.run_once()
        return

    # graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    task = loop.create_task(trader.run())

    def _shutdown(sig_name: str) -> None:
        print(f"\n  [{sig_name}] shutting down…")
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig.name)

    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║          smart-trader  ·  paper trading             ║")
    print("║   Ctrl+C to stop gracefully                         ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    asyncio.run(main())
