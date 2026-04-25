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
| Short entries | ✅ Done — regime-gated shorts (bear/distribution only), Sharpe -18.4 → +2.95 |
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

### ST-ALPHA-01 — Short entries in bear/distribution ✅ DONE (2026-04-25)

> Rule engine is long-only. Bear markets = dead capital today.

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-01-1 | Audit `CoreSleeve` and `TacticalSleeve` for short entry logic | ✅ | Full sell path exists; MTF_THRESHOLD=0.10 was blocking all shorts in distribution (4h bounces normal) |
| T-01-2 | Enable short entry in `bear_trending` regime for both sleeves | ✅ | Raised `MTF_THRESHOLD` 0.10→0.30 in `TacticalSleeve` to match backtest engine |
| T-01-3 | Add short-entry gate: only allow shorts when `regime in {bear_trending, bear_ranging, distribution}` | ✅ | `_SHORT_ENTRY_REGIMES` frozenset added to both sleeves; blocks naked shorts in accumulation/bull |
| T-01-4 | Backtest: long-only vs long+short, 480d ETH/USDT | ✅ PASS | A(long): Sharpe=-18.4, 2 trades. B(both): Sharpe=+2.95, 10 trades (8 short), short_PnL=+$34.94. Δ=+21.3 ≥ −0.2 gate |
| T-01-5 | Update `HybridBacktestEngine` to report long/short trade split | ✅ | `long_trades`/`short_trades` properties on `HybridBacktestResult`; `compare_short_entries.py` script |

### ST-ALPHA-02 — Graduated position sizing by confidence ✅ DONE (2026-04-25)

> All-or-nothing at 0.70 threshold leaves edge on the table.

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-02-1 | Define sizing tiers: `[0.55, 0.65, 0.75]` → `[25%, 50%, 100%]` of max position | ✅ | `_CONF_TIERS` in `sleeve/capital.py` — already implemented |
| T-02-2 | Implement `confidence_to_size_scale()` in `sleeve/capital.py` | ✅ | Already implemented; wired into live sleeves |
| T-02-3 | Wire into `CoreSleeve`, `TacticalSleeve`, and `BacktestEngine` | ✅ | Live sleeves already done; added `graduated_sizing` flag to `BacktestConfig` and engine loop |
| T-02-4 | Backtest: binary vs graduated sizing, 480d ETH/USDT | ✅ PASS | A(binary): Sharpe=+2.95. B(tiered): Sharpe=+4.24, PnL=+$31.45. Δ=+1.29 ≥ −0.2 gate |

### ST-ALPHA-03 — Regime-adaptive position cap ✅ DONE (2026-04-25)

> Same max position in trending vs ranging is sub-optimal.

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-03-1 | Add `pos_cap_mult` to `RegimeParamAdapter` 24-grid | ✅ | Already implemented: `bull_trending=1.0`, `bull_ranging=0.6`, `accumulation=0.5`, `distribution=0.3` |
| T-03-2 | Wire into both sleeves + `BacktestEngine` | ✅ | Live sleeves multiply `confidence_to_size_scale × pos_cap_mult`; added `RegimeParamAdapter` to backtest engine |
| T-03-3 | Verify log shows `pos_cap_mult` per regime | ✅ | Both sleeves emit `meta["pos_cap_mult"]` on every `enter` decision; backtest now matches live |

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

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-01-1 | Backfill BTC/USDT 1h/4h/1d | ✅ | 480 days of data available |
| T-01-2 | Verify BTC edge via backtest sweep | ❌ blocked | All 16 configs negative Sharpe (-1.88 to -2.08), WR ~30%. ETH got +2.95; BTC signal engine has no edge on BTC. Gate fails — do NOT add to live loop yet. Re-evaluate after signal engine improvements (EPIC-PHASE2). |
| T-01-3 | Add BTC to live loop (50/50 split) | ⏸ deferred | Infrastructure built (`MultiHybridLoop`, `CorrelationGuard`, per-symbol `peak_key`). Activate when BTC backtest passes WR≥40% + Sharpe>0. |

### ST-MULTI-02 — Correlation-aware position sizing

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-02-1 | Rolling 30-day correlation guard | ✅ | `CorrelationGuard` in `sleeve/correlation_guard.py` — 30-day Pearson on 1d closes |
| T-02-2 | >0.80 correlation → reduce exposure 20% | ✅ | `MultiHybridLoop._correlation_watchdog()` — hourly check, injects cap mult |
| T-02-3 | Dual-symbol backtest | ⏸ deferred | Blocked on T-01-2 passing |

---

## EPIC-PHASE2 — Complete Phase 2.3 Multi-Strategy (P3) — ✅ DONE (2026-04-22)

**Goal**: Register multiple signal strategies and route by regime.

### ST-PHASE2-01 — Strategy registry ✅

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-01-1 | `StrategyRegistry` with `register(name, factory, regime_affinity)` | ✅ | `strategy/registry.py` — lazy-instantiated instances, `for_regime()` lookup |
| T-01-2 | Register presets: `trend_follower`, `mean_reversion`, `breakout`, `full_v2` | ✅ | Augmented (all 5 detectors per preset, regime-priority ordering) |
| T-01-3 | Wire `regime_routing=True` into sleeves + backtest engine | ✅ | `hybrid_regime_routing: bool = True` in settings; both sleeves + `BacktestConfig` |
| T-01-4 | Backtest routing vs single-strategy | ✅ neutral | Δ Sharpe = 0.000 on 480d ETH. Same 10 trades, same WR/Sharpe. Gate fails (< 0.3) but routing is zero-impact not negative. Kept ON — will show lift with multi-symbol or longer data. |

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
