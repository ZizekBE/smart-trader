"""Sleeve data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from smart_trader.risk.models import Portfolio, RiskDecision
from smart_trader.strategy.signals.models import SignalEvent


class SleeveId(StrEnum):
    CORE     = "core"       # long-term positions
    TACTICAL = "tactical"   # short-term positions


@dataclass
class SleeveDecision:
    """Output from a sleeve's evaluate() call."""
    sleeve:   SleeveId
    signal:   Optional[SignalEvent]
    decision: Optional[RiskDecision]
    portfolio: Portfolio             # sleeve-scoped portfolio used for sizing
    tick_meta: dict = field(default_factory=dict)  # diagnostic info for signal log


@dataclass
class SleeveState:
    """Runtime state tracked per sleeve."""
    sl_cooldown:   dict[str, int] = field(default_factory=lambda: {"buy": 0, "sell": 0})
    vol_regime_at_entry: dict[str, str] = field(default_factory=lambda: {"buy": "", "sell": ""})
    cur_vol_regime: str = ""
