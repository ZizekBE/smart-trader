"""
Central configuration — loaded once at startup via pydantic-settings.

Environment variables are populated from (in order):

1. ``configs/envs/.env`` — non-secret defaults (optional)
2. ``configs/secrets/.env`` — credentials; overrides (optional, gitignored)

See ``configs/envs/.env.example`` and ``configs/secrets/.env.example``.
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from smart_trader.core.env_files import load_repo_env_files


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE  = "live"


class LogLevel(StrEnum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    db_host:     str = "localhost"
    db_port:     int = 5432
    db_user:     str = "trader"
    db_password: SecretStr = Field(default="changeme")
    db_name:     str = "smart_trader"

    @property
    def db_url(self) -> str:
        pw = self.db_password.get_secret_value()
        return f"postgresql+asyncpg://{self.db_user}:{pw}@{self.db_host}:{self.db_port}/{self.db_name}"

    # ── Redis ─────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db:   int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Exchange ──────────────────────────────────────────────
    default_exchange:   str         = "gateio"
    symbols:            list[str]   = ["BTC/USDT", "ETH/USDT"]
    timeframes:         list[str]   = ["1m", "1h", "4h", "1d", "1w"]
    trading_mode:       TradingMode = TradingMode.PAPER

    gateio_api_key:     SecretStr = Field(default="")
    gateio_api_secret:  SecretStr = Field(default="")

    # kept for future multi-exchange support
    binance_api_key:    SecretStr = Field(default="")
    binance_api_secret: SecretStr = Field(default="")
    okx_api_key:        SecretStr = Field(default="")
    okx_api_secret:     SecretStr = Field(default="")
    okx_passphrase:     SecretStr = Field(default="")

    # ── Risk limits ───────────────────────────────────────────
    max_position_size_pct: float = 0.05   # 5 % per trade
    max_daily_loss_pct:    float = 0.02   # 2 % daily circuit breaker
    max_drawdown_pct:      float = 0.10   # 10 % halt

    # ── Signal strategy ───────────────────────────────────────
    strategy_version:       str   = "v2"   # set to "v1" in .env to roll back
    confidence_threshold:   float = 0.65
    model_artifacts_path:   str   = "./models"

    # ── Risk param overrides (from Bayesian optimisation) ─────
    # Leave unset to use PositionSizer mode defaults.
    opt_atr_mult:              float | None = None
    opt_rr_ratio:              float | None = None
    opt_kelly_frac:            float | None = None
    opt_trailing_atr_mult:     float | None = None
    opt_trailing_trigger_pct:  float | None = None
    opt_breakeven_trigger_pct: float | None = None
    opt_pullback_atr_frac:     float | None = None
    opt_sl_cooldown_bars:      int   | None = None
    opt_min_conf_high_vol:     float | None = None

    # ── Bear market filter ────────────────────────────────────
    bear_filter_bars:      int   = 3      # consecutive bearish 1d bars to block long entries
    bear_filter_threshold: float = -0.15  # 1d direction threshold (< this = bearish day)

    # ── Hybrid strategy ───────────────────────────────────────
    hybrid_long_budget_pct:   float = 0.60
    hybrid_short_budget_pct:  float = 0.30
    hybrid_long_signal_tf:    str   = "4h"
    hybrid_long_trend_tf:     str   = "1d"
    hybrid_long_min_conf:     float = 0.55  # ST-ALPHA-04: sweep recommended (Sharpe+2.95, WR 40%)
    hybrid_long_max_pos_pct:  float = 0.30
    hybrid_long_atr_mult:     float = 3.0
    hybrid_long_rr_ratio:     float = 3.0
    hybrid_short_signal_tf:   str   = "1h"
    hybrid_short_trend_tf:    str   = "1d"
    hybrid_short_mid_tf:      str   = "4h"
    hybrid_short_min_conf:    float = 0.60  # ST-ALPHA-04: sweep recommended (Sharpe+2.95, WR 40%)
    hybrid_short_max_pos_pct: float = 0.10
    hybrid_short_time_stop_h: int   = 24
    # session filter (tactical sleeve only)
    hybrid_session_filter:     bool = True
    hybrid_session_start_utc:  int  = 7
    hybrid_session_end_utc:    int  = 22
    # partial profit taking (both sleeves)
    hybrid_partial_tp:         bool  = True
    hybrid_partial_tp_r:       float = 1.0
    hybrid_partial_tp_pct:     float = 0.50
    # Phase 2.3 regime-based strategy routing
    hybrid_regime_routing:     bool  = True

    # ── Rust engine ───────────────────────────────────────────
    rust_engine_host: str = "localhost"
    rust_engine_port: int = 8080

    # ── Observability ─────────────────────────────────────────
    log_level:        LogLevel = LogLevel.INFO
    prometheus_port:  int      = 9100


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings — safe to call anywhere."""
    load_repo_env_files()
    return Settings()
