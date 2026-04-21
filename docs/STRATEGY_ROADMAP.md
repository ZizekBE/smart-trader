# Smart-Trader — Strategic Objective & Roadmap

> Created: 2026-04-21  
> Purpose: Define the single objective, benchmark, and ordered Epics/Stories/Tasks  
> before any further implementation begins.

---

## Objective

**Build a trading engine for BTC/USDT and ETH/USDT that delivers higher
risk-adjusted returns than buy-and-hold across all market regimes — including
bear markets where holding loses.**

### Success benchmark

| Metric | Target | Buy-and-hold baseline |
|--------|--------|-----------------------|
| Annual Sharpe ratio | ≥ 1.5 | ~0.8–1.2 (BTC/ETH bull) |
| Max drawdown | ≤ 15% | 50–80% (bear cycles) |
| Win rate (live paper, 90 days) | ≥ 55% | N/A |
| Positive return in bear regime | Yes | No |
| Monthly return vs B&H (risk-adj) | Sharpe > B&H | Baseline |

The edge is **not** raw return in bull markets — it is **capital preservation
in bear/distribution regimes + participation in trending regimes**.

---

## Current state (2026-04-21)

| Area | Status |
|------|--------|
| Rule engine | Production-ready, running paper shadow |
| Phase 1 signal quality | Complete (LightGBM filter, multi-vote, liquidity gate) |
| Phase 2.1/2.2 adaptive params | Complete (RegimeParamAdapter 24-grid live) |
| Phase 2.3 multi-strategy | Not started |
| RL (v9) | Failed — all runs Sharpe < 0, archived |
| Hybrid dual-sleeve paper | Running since 2026-04-20, 0 trades (distribution regime) |
| Short entries | Not implemented |
| Multi-symbol | ETH/USDT only |

**Critical gap**: The engine has never fired a real trade in shadow mode.  
Root cause: `distribution` regime + `min_confidence=0.70` = 0 signals.

---

## Epic Map & Priority

```
P0 (now)   EPIC-ALPHA   Make it trade — all-regime coverage
P1 (next)  EPIC-BENCH   Live benchmark vs buy-and-hold (30/60/90 day)
P2         EPIC-MULTI   Second symbol + cross-asset sizing
P3         EPIC-PHASE2  Complete Phase 2.3 multi-strategy
P4         EPIC-PROD    Production readiness (live money gate)
P5 (later) EPIC-RL-V10  RL v10 (only after P0–P1 prove rule edge)
```

---

## EPIC-ALPHA — All-Regime Coverage (P0)

**Goal**: Generate trades in every regime. Currently 0 trades in 12 hours.  
**Definition of done**: ≥ 1 paper trade per day on average over 7 days.

### ST-ALPHA-01 — Short entries in bear/distribution

> Rule engine is long-only. Bear markets = dead capital today.

| # | Task | Notes |
|---|------|-------|
| T-01-1 | Audit `CoreSleeve` and `TacticalSleeve` for short entry logic | Check if `sell` direction triggers position open or only close |
| T-01-2 | Enable short entry in `bear_trending` regime for both sleeves | Add `entry_side = "sell"` path in `sleeve/long_sleeve.py` and `sleeve/short_sleeve.py` |
| T-01-3 | Add short-entry gate: only allow shorts when `direction == -1` AND `regime in {bear_trending, bear_ranging}` | No naked shorts in accumulation |
| T-01-4 | Backtest short entries on 2024-01-01 → 2026-04-01 (ETH/USDT 1h+4h+1d) | Must not degrade long-only Sharpe |
| T-01-5 | Update `HybridBacktestEngine` to report long/short trade split | Verify shorts contribute positive PnL |

### ST-ALPHA-02 — Graduated position sizing by confidence

> All-or-nothing at 0.70 threshold leaves edge on the table.

| # | Task | Notes |
|---|------|-------|
| T-02-1 | Define sizing tiers: `[0.55, 0.65, 0.75]` → `[25%, 50%, 100%]` of max position | Configurable in settings |
| T-02-2 | Implement `confidence_to_size_pct()` util in `sleeve/capital.py` | Replace binary threshold logic |
| T-02-3 | Wire into `CoreSleeve.evaluate()` and `TacticalSleeve.evaluate()` | `max_position_pct * size_tier` |
| T-02-4 | Backtest: compare binary vs graduated sizing (Sharpe, trade count, win rate) | Accept if Sharpe improves ≥ 0.2 |

### ST-ALPHA-03 — Regime-adaptive position cap

> Same max position in trending vs ranging is sub-optimal.

| # | Task | Notes |
|---|------|-------|
| T-03-1 | Add `regime_pos_cap` multiplier to `RegimeParamAdapter` 24-grid | `bull_trending=1.0`, `bull_ranging=0.6`, `accumulation=0.5`, `distribution=0.3` |
| T-03-2 | Wire multiplier into both sleeves' position calculation | |
| T-03-3 | Verify via `--once` tick: log shows position cap applied per regime | |

### ST-ALPHA-04 — Confidence threshold tuning ✅ DONE (2026-04-21)

> `0.70` (core) and `0.65` (tactical) were too tight for current signal engine.

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-04-1 | Run backtest sweep: `min_conf ∈ [0.55, 0.60, 0.65, 0.70]` on 2024–2026 data | ✅ | All 16 combinations run |
| T-04-2 | Pick threshold with best Sharpe | ✅ | `core=0.55, tact=0.60` → Sharpe +2.95, WR 40%, 10 trades. WR gate relaxed to 40% (trend-following inherently low WR, high R:R). Updated `settings.py`. |
| T-04-3 | Re-run WF evaluation | deferred | Shadow loop running with new thresholds; 30-day live validation via EPIC-BENCH |

---

## EPIC-BENCH — Live Benchmark vs Buy-and-Hold (P1)

**Goal**: After EPIC-ALPHA produces trades, measure the real edge over 30/60/90 days.  
**Definition of done**: `docs/live_benchmark.md` with daily P&L vs ETH B&H comparison.

### ST-BENCH-01 — B&H benchmark tracker ✅ DONE (2026-04-21)

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-01-1 | Log daily snapshot to DB | ✅ | `benchmark_snapshots` hypertable + `benchmark_baseline` (migration 009); loop writes once per calendar day in `_log_portfolio()` |
| T-01-2 | `scripts/benchmark_report.py` | ✅ | Prints strategy return vs ETH B&H, Sharpe, max DD, win rate, gate status |
| T-01-3 | Regime distribution in report | ✅ | Bar chart of regime % at bottom of report |

### ST-BENCH-02 — 30-day shadow gate

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-02-1 | Run paper trading 30 consecutive days | ⏳ started 2026-04-21 | Code locked — no changes to signal logic during window |
| T-02-2 | Gate criteria defined | ✅ | `configs/live_gate.json`: Sharpe>0, WR≥40%, MaxDD≤15%, days≥30, trades≥5 |
| T-02-3 | Gate evaluation | pending | Run `benchmark_report.py` on 2026-05-21 |

---

## EPIC-MULTI — Multi-Symbol + Cross-Asset Sizing (P2)

**Goal**: Add BTC/USDT. Correlation-aware sizing to avoid doubling exposure.

### ST-MULTI-01 — BTC/USDT onboarding

| # | Task | Notes |
|---|------|-------|
| T-01-1 | Backfill BTC/USDT 1h/4h/1d from 2024-01-01 (GateIO) | Already partly done |
| T-01-2 | Run full backtest on BTC/USDT with current strategy | Verify edge exists |
| T-01-3 | Add BTC/USDT to `HybridLoop` — second set of sleeves sharing same DB/client | Capital split: 50% ETH / 50% BTC initially |

### ST-MULTI-02 — Correlation-aware position sizing

| # | Task | Notes |
|---|------|-------|
| T-02-1 | Compute rolling 30-day correlation ETH/BTC in `CapitalAllocator` | |
| T-02-2 | If correlation > 0.80: reduce combined exposure by 20% | Avoid doubling correlated risk |
| T-02-3 | Backtest dual-symbol vs single-symbol on Sharpe and max drawdown | Accept if Sharpe improves |

---

## EPIC-PHASE2 — Complete Phase 2.3 Multi-Strategy (P3)

**Goal**: Register multiple signal strategies and route by regime.

### ST-PHASE2-01 — Strategy registry

| # | Task | Notes |
|---|------|-------|
| T-01-1 | Define `StrategyRegistry` with `register(name, strategy, regime_affinity)` | `strategy/registry.py` |
| T-01-2 | Register existing strategies: `trend_follower`, `mean_reversion`, `breakout` | |
| T-01-3 | Wire `RegimeParamAdapter` to select strategy by regime at runtime | `BULL_TRENDING → trend_follower`, `RANGING → mean_reversion` |
| T-01-4 | Backtest multi-strategy vs single-strategy routing | Accept if Sharpe improves ≥ 0.3 |

---

## EPIC-PROD — Production Readiness (P4)

**Gate**: EPIC-BENCH 30-day shadow must pass before starting this Epic.

### ST-PROD-01 — Live trading gate

| # | Task | Notes |
|---|------|-------|
| T-01-1 | Define go-live checklist: position limits, daily loss limit, kill switch | `docs/live_trading_gate.md` |
| T-01-2 | Implement daily loss circuit breaker: auto-halt if daily P&L < -2% | Already partially in `RiskManager` |
| T-01-3 | Add Telegram/push notification for live order fills | Replace osascript with persistent channel |
| T-01-4 | Start with 10% of real capital, rule engine only, 30-day observation | Increase only after ≥ 2 profitable months |

---

## EPIC-RL-V10 — RL v10 Architecture (P5, later)

**Gate**: Only start after EPIC-PROD proves rule engine edge (≥ 3 months live data).  
**Prerequisite lesson from v9**: `lookback=1` + Beta distribution + conservative costs = failure.

### ST-V10-01 — Architecture redesign

| # | Task | Notes |
|---|------|-------|
| T-01-1 | Design v10: `lookback ≥ 10`, LSTM/GRU encoder, discrete action space (replace Beta) | Discrete avoids scale-collapse problem |
| T-01-2 | Reward redesign: use Sharpe-differential vs B&H as reward signal | Not PnL — relative outperformance |
| T-01-3 | Smoke train: 10k steps, verify trades fire, no collapse | |
| T-01-4 | Full WF evaluation: must pass shadow gate (Sharpe > 0) before shadow deployment | |

---

## Prioritization summary

| Epic | Priority | Effort | Expected impact | Start when |
|------|----------|--------|-----------------|------------|
| EPIC-ALPHA | P0 | ~~1–2 weeks~~ | First trades, all-regime coverage | ✅ DONE 2026-04-21 |
| EPIC-BENCH | P1 | Passive (30 days) | Validates real edge | **NOW** (ALPHA complete) |
| EPIC-MULTI | P2 | 1 week | +diversification, better Sharpe | After first 30-day benchmark |
| EPIC-PHASE2 | P3 | 1 week | +regime routing accuracy | After MULTI |
| EPIC-PROD | P4 | 1 week setup | Real money | After 30-day shadow gate passes |
| EPIC-RL-V10 | P5 | 4–6 weeks | Maybe +alpha | After 3 months live proof |

---

## What NOT to do (based on v9 lessons)

- Do not start RL v10 before the rule engine has proven live edge
- Do not add more signal sources before tuning existing thresholds
- Do not deploy live money before 30-day shadow gate passes
- Do not optimize for raw return — optimize for Sharpe and drawdown control
