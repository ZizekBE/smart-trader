"""Shared in-process state between TradingLoop and the API server."""
from __future__ import annotations

from collections import deque
from typing import Any

# Circular buffer of per-tick analysis records.
# Populated by TradingLoop via callback; read by GET /api/trader/signals.
signal_log: deque[dict[str, Any]] = deque(maxlen=100)
