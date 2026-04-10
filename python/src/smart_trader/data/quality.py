"""Data quality checks — gap detection and outlier filtering.

Validates candle data integrity after backfill:
  - Gap detection: finds missing bars in the time series
  - Outlier detection: flags bars with extreme price moves or suspicious volume
  - Duplicate detection: ensures no duplicate timestamps

Usage::

    checker = DataQualityChecker()
    report = await checker.check("BTC/USDT", "gateio", "1m",
                                  since=datetime(2024, 4, 1, ...),
                                  until=datetime(2026, 4, 1, ...))
    print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import structlog

from smart_trader.data.ingestion.gateio_client import TIMEFRAME_MS
from smart_trader.data.storage.candle_repo import CandleRepository
from smart_trader.data.storage.database import get_session_factory

log = structlog.get_logger(__name__)


@dataclass
class Gap:
    start: datetime
    end: datetime
    missing_bars: int


@dataclass
class Outlier:
    time: datetime
    field: str       # "close", "volume", etc.
    value: float
    z_score: float
    reason: str


@dataclass
class QualityReport:
    symbol: str
    exchange: str
    timeframe: str
    total_bars: int = 0
    expected_bars: int = 0
    coverage_pct: float = 0.0
    gaps: list[Gap] = field(default_factory=list)
    outliers: list[Outlier] = field(default_factory=list)
    duplicates: int = 0
    first_bar: Optional[datetime] = None
    last_bar: Optional[datetime] = None

    @property
    def is_healthy(self) -> bool:
        return (
            self.coverage_pct >= 99.0
            and len(self.gaps) == 0
            and len(self.outliers) <= self.total_bars * 0.001
        )

    def summary(self) -> str:
        health = "HEALTHY" if self.is_healthy else "ISSUES FOUND"
        lines = [
            f"\n{'─'*56}",
            f"  Quality Report: {self.symbol} / {self.exchange} / {self.timeframe}  [{health}]",
            f"{'─'*56}",
            f"  Range:      {self.first_bar}  →  {self.last_bar}",
            f"  Bars:       {self.total_bars:,} / {self.expected_bars:,} expected",
            f"  Coverage:   {self.coverage_pct:.2f}%",
            f"  Gaps:       {len(self.gaps)}",
            f"  Outliers:   {len(self.outliers)}",
            f"  Duplicates: {self.duplicates}",
        ]
        if self.gaps:
            lines.append(f"  {'─'*50}")
            lines.append(f"  Top gaps (by missing bars):")
            for g in sorted(self.gaps, key=lambda g: g.missing_bars, reverse=True)[:10]:
                lines.append(f"    {g.start} → {g.end}  ({g.missing_bars} bars)")
        if self.outliers:
            lines.append(f"  {'─'*50}")
            lines.append(f"  Outliers (top 10 by z-score):")
            for o in sorted(self.outliers, key=lambda o: abs(o.z_score), reverse=True)[:10]:
                lines.append(f"    {o.time}  {o.field}={o.value:.4f}  z={o.z_score:.1f}  {o.reason}")
        lines.append(f"{'─'*56}")
        return "\n".join(lines)


class DataQualityChecker:
    """Validates candle data integrity in the database."""

    def __init__(
        self,
        price_z_threshold: float = 8.0,
        volume_z_threshold: float = 10.0,
        min_gap_bars: int = 2,
    ) -> None:
        self._price_z = price_z_threshold
        self._volume_z = volume_z_threshold
        self._min_gap = min_gap_bars
        self._factory = get_session_factory()

    async def check(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> QualityReport:
        """Run all quality checks on stored candle data."""
        until = until or datetime.now(timezone.utc)
        since = since or (until - timedelta(days=365))

        async with self._factory() as session:
            repo = CandleRepository(session)
            candles = await repo.get_range(symbol, exchange, timeframe, since, until)

        report = QualityReport(
            symbol=symbol, exchange=exchange, timeframe=timeframe,
        )

        if not candles:
            log.warning("no_data_for_quality_check", symbol=symbol, tf=timeframe)
            return report

        tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
        tf_delta = timedelta(milliseconds=tf_ms)

        report.total_bars = len(candles)
        report.first_bar = candles[0].time
        report.last_bar = candles[-1].time
        total_span = (report.last_bar - report.first_bar).total_seconds() * 1000
        report.expected_bars = int(total_span / tf_ms) + 1
        report.coverage_pct = (
            report.total_bars / max(report.expected_bars, 1) * 100
        )

        # gap detection
        report.gaps = self._detect_gaps(candles, tf_delta)

        # duplicate detection
        times = [c.time for c in candles]
        report.duplicates = len(times) - len(set(times))

        # outlier detection
        report.outliers = self._detect_outliers(candles)

        return report

    def _detect_gaps(self, candles, tf_delta: timedelta) -> list[Gap]:
        """Find gaps where consecutive candle timestamps differ by more than 1 bar."""
        gaps = []
        for i in range(1, len(candles)):
            diff = candles[i].time - candles[i - 1].time
            expected_diff = tf_delta
            # allow 10% tolerance for DST / exchange maintenance
            if diff > expected_diff * 1.1:
                missing = int(diff / expected_diff) - 1
                if missing >= self._min_gap:
                    gaps.append(Gap(
                        start=candles[i - 1].time,
                        end=candles[i].time,
                        missing_bars=missing,
                    ))
        return gaps

    def _detect_outliers(self, candles) -> list[Outlier]:
        """Flag bars with extreme price moves or suspicious volume."""
        if len(candles) < 20:
            return []

        closes = np.array([float(c.close) for c in candles])
        volumes = np.array([float(c.volume) for c in candles])
        highs = np.array([float(c.high) for c in candles])
        lows = np.array([float(c.low) for c in candles])

        # price return z-scores
        returns = np.diff(closes) / (closes[:-1] + 1e-9)
        ret_mean, ret_std = np.mean(returns), np.std(returns) + 1e-9

        # volume z-scores (rolling window)
        vol_mean = np.convolve(volumes, np.ones(20) / 20, mode="same")
        vol_std = np.array([
            np.std(volumes[max(0, i - 20):i + 1]) + 1e-9
            for i in range(len(volumes))
        ])

        outliers = []
        for i in range(1, len(candles)):
            c = candles[i]

            # extreme return
            ret_z = (returns[i - 1] - ret_mean) / ret_std
            if abs(ret_z) > self._price_z:
                outliers.append(Outlier(
                    time=c.time, field="return",
                    value=float(returns[i - 1]),
                    z_score=float(ret_z),
                    reason=f"return {returns[i-1]:.4%} is {abs(ret_z):.1f}σ from mean",
                ))

            # extreme volume
            vol_z = (volumes[i] - vol_mean[i]) / vol_std[i]
            if vol_z > self._volume_z:
                outliers.append(Outlier(
                    time=c.time, field="volume",
                    value=float(volumes[i]),
                    z_score=float(vol_z),
                    reason=f"volume {volumes[i]:.2f} is {vol_z:.1f}σ above mean",
                ))

            # high < low (data corruption)
            if highs[i] < lows[i]:
                outliers.append(Outlier(
                    time=c.time, field="high_low",
                    value=float(highs[i]),
                    z_score=0.0,
                    reason=f"high ({highs[i]}) < low ({lows[i]})",
                ))

            # zero or negative price
            if closes[i] <= 0:
                outliers.append(Outlier(
                    time=c.time, field="close",
                    value=float(closes[i]),
                    z_score=0.0,
                    reason="zero or negative close price",
                ))

        return outliers


async def run_quality_checks(
    symbols: list[str],
    exchange: str,
    timeframes: list[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[QualityReport]:
    """Run quality checks across multiple symbols and timeframes."""
    checker = DataQualityChecker()
    reports = []
    for symbol in symbols:
        for tf in timeframes:
            report = await checker.check(symbol, exchange, tf, since, until)
            reports.append(report)
            log.info(
                "quality_check",
                symbol=symbol, tf=tf,
                bars=report.total_bars,
                coverage=f"{report.coverage_pct:.2f}%",
                gaps=len(report.gaps),
                outliers=len(report.outliers),
                healthy=report.is_healthy,
            )
    return reports
