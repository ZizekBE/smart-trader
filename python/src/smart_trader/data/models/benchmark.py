"""ORM models for EPIC-BENCH live benchmark tracking."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class BenchmarkSnapshot(Base):
    __tablename__ = "benchmark_snapshots"

    ts:              Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol:          Mapped[str]      = mapped_column(String(20),  primary_key=True)
    portfolio_total: Mapped[float]    = mapped_column(Numeric(20, 8), nullable=False)
    cash:            Mapped[float]    = mapped_column(Numeric(20, 8), nullable=False)
    bh_price:        Mapped[float]    = mapped_column(Numeric(20, 8), nullable=False)
    regime:          Mapped[str]      = mapped_column(String(40),  nullable=True)
    positions:       Mapped[int]      = mapped_column(Integer,     nullable=False, default=0)


class BenchmarkBaseline(Base):
    __tablename__ = "benchmark_baseline"

    symbol:        Mapped[str]      = mapped_column(String(20),  primary_key=True)
    start_price:   Mapped[float]    = mapped_column(Numeric(20, 8), nullable=False)
    start_capital: Mapped[float]    = mapped_column(Numeric(20, 8), nullable=False)
    start_ts:      Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
