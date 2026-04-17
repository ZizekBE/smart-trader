"""Futures data coverage check — funding_rate and open_interest.

Queries TimescaleDB to report row counts, date ranges, and approximate
coverage percentage for funding_rates and open_interest tables per symbol.

Usage::

    cd python && uv run python scripts/check_futures_coverage.py
    uv run python scripts/check_futures_coverage.py --symbols BTC/USDT ETH/USDT --exchange binance
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import structlog  # noqa: E402

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check futures data coverage in DB")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="Minimum coverage fraction to pass gate (default: 0.80)")
    return p.parse_args()


async def check_coverage(
    symbols: list[str],
    exchange: str,
    threshold: float,
) -> dict:
    from sqlalchemy import text
    from smart_trader.data.storage.database import get_engine

    engine = get_engine()
    results: dict[str, dict] = {}

    async with engine.connect() as conn:
        for symbol in symbols:
            sym_result: dict = {"symbol": symbol, "exchange": exchange}

            # ── funding_rates coverage ──────────────────────────────────────
            # Expected interval: every 8 hours  →  3 rows/day
            fr_sql = text("""
                SELECT
                    COUNT(*)                          AS row_count,
                    MIN(time)                         AS min_time,
                    MAX(time)                         AS max_time,
                    EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) / 3600.0
                                                      AS span_hours
                FROM funding_rates
                WHERE symbol = :sym AND exchange = :ex
            """)
            fr_row = (await conn.execute(fr_sql, {"sym": symbol, "ex": exchange})).fetchone()

            if fr_row and fr_row.row_count > 0:
                span_h = float(fr_row.span_hours or 0)
                expected_rows = max(1, span_h / 8.0)          # one settlement per 8 h
                coverage = min(1.0, fr_row.row_count / expected_rows)
                sym_result["funding_rate"] = {
                    "row_count": int(fr_row.row_count),
                    "min_time": str(fr_row.min_time),
                    "max_time": str(fr_row.max_time),
                    "span_days": round(span_h / 24, 1),
                    "expected_rows": int(expected_rows),
                    "coverage_pct": round(coverage * 100, 1),
                    "pass": coverage >= threshold,
                }
            else:
                sym_result["funding_rate"] = {"row_count": 0, "coverage_pct": 0.0, "pass": False}

            # ── open_interest coverage ──────────────────────────────────────
            # OI is snapshot-only; check distinct-day coverage vs candle span.
            oi_sql = text("""
                SELECT
                    COUNT(*)                          AS row_count,
                    COUNT(DISTINCT time::date)        AS distinct_days,
                    MIN(time)                         AS min_time,
                    MAX(time)                         AS max_time,
                    EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) / 86400.0
                                                      AS span_days
                FROM open_interest
                WHERE symbol = :sym AND exchange = :ex
            """)
            oi_row = (await conn.execute(oi_sql, {"sym": symbol, "ex": exchange})).fetchone()

            if oi_row and oi_row.row_count > 0:
                span_d = float(oi_row.span_days or 0)
                expected_days = max(1, span_d)
                day_coverage = min(1.0, int(oi_row.distinct_days) / expected_days)
                sym_result["open_interest"] = {
                    "row_count": int(oi_row.row_count),
                    "distinct_days": int(oi_row.distinct_days),
                    "min_time": str(oi_row.min_time),
                    "max_time": str(oi_row.max_time),
                    "span_days": round(span_d, 1),
                    "day_coverage_pct": round(day_coverage * 100, 1),
                    "pass": day_coverage >= threshold,
                }
            else:
                sym_result["open_interest"] = {"row_count": 0, "day_coverage_pct": 0.0, "pass": False}

            # ── candle baseline (for comparison) ───────────────────────────
            candle_sql = text("""
                SELECT
                    COUNT(*)    AS row_count,
                    MIN(time)   AS min_time,
                    MAX(time)   AS max_time
                FROM candles
                WHERE symbol = :sym AND exchange = :ex AND timeframe = '1h'
            """)
            c_row = (await conn.execute(candle_sql, {"sym": symbol, "ex": exchange})).fetchone()
            if c_row and c_row.row_count > 0:
                sym_result["candle_1h"] = {
                    "row_count": int(c_row.row_count),
                    "min_time": str(c_row.min_time),
                    "max_time": str(c_row.max_time),
                }
            else:
                sym_result["candle_1h"] = {"row_count": 0}

            results[symbol] = sym_result

    return results


def print_report(results: dict, threshold: float) -> bool:
    """Pretty-print coverage report. Returns True if all symbols pass."""
    w = 70
    print(f"\n{'═'*w}")
    print(f"  Futures Data Coverage Report  (gate: {threshold*100:.0f}%)")
    print(f"{'═'*w}")

    all_pass = True

    for symbol, data in results.items():
        print(f"\n  Symbol: {symbol}  |  Exchange: {data['exchange']}")
        print(f"  {'─'*64}")

        # candle baseline
        c = data.get("candle_1h", {})
        print(f"  Candle 1h rows:   {c.get('row_count', 0):>8,}")
        if c.get("min_time"):
            print(f"  Candle range:     {c['min_time'][:19]}  →  {c['max_time'][:19]}")

        # funding rate
        fr = data.get("funding_rate", {})
        status = "PASS" if fr.get("pass") else "FAIL"
        print(f"\n  Funding Rate:   [{status}]")
        print(f"    Rows:         {fr.get('row_count', 0):>8,}")
        if fr.get("span_days"):
            print(f"    Span:         {fr['span_days']} days")
            print(f"    Expected:     {fr.get('expected_rows', 0):>8,}  (3 per day × 8 h)")
            print(f"    Coverage:     {fr.get('coverage_pct', 0):.1f}%")
        if fr.get("min_time"):
            print(f"    Range:        {fr['min_time'][:19]}  →  {fr['max_time'][:19]}")
        if not fr.get("pass"):
            all_pass = False

        # open interest
        oi = data.get("open_interest", {})
        status = "PASS" if oi.get("pass") else "FAIL"
        print(f"\n  Open Interest:  [{status}]")
        print(f"    Rows:         {oi.get('row_count', 0):>8,}")
        if oi.get("span_days"):
            print(f"    Span:         {oi['span_days']} days")
            print(f"    Distinct days:{oi.get('distinct_days', 0):>8,}")
            print(f"    Day coverage: {oi.get('day_coverage_pct', 0):.1f}%")
        if oi.get("min_time"):
            print(f"    Range:        {oi['min_time'][:19]}  →  {oi['max_time'][:19]}")
        if not oi.get("pass"):
            all_pass = False
            print(f"    NOTE: OI backfill fetches current snapshot only — historical")
            print(f"          coverage accumulates during live ingestion runs.")

    print(f"\n{'═'*w}")
    verdict = "GATE PASSED" if all_pass else "GATE FAILED — see FAIL items above"
    print(f"  {verdict}")
    print(f"{'═'*w}\n")

    return all_pass


async def main() -> None:
    args = parse_args()
    log.info("coverage_check_start", symbols=args.symbols, exchange=args.exchange)
    results = await check_coverage(args.symbols, args.exchange, args.threshold)
    passed = print_report(results, args.threshold)

    # Structured log for CI consumption
    log.info("coverage_gate", passed=passed, threshold=args.threshold)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
