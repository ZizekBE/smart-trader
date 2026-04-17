# Phase 2.3 — Multi-strategy routing: walk-forward comparison

Dataset : `2022-03-01` → `2026-04-11`  
Rows    : 17,373  |  Folds: 20 × 7d

## Summary — aggregated across all folds

| Strategy | Signals/fold | Win rate | Mean ret | Sharpe (μ) | Sharpe (σ) | +folds |
|----------|-------------|---------|---------|-----------|-----------|--------|
| v2 (all det.) | 83.2 | 50.4% | -0.071% | -0.77 | 4.65 | 45.0% |
| trend_follower | 32.0 | 42.0% | -0.272% | -3.17 | 5.88 | 40.0% |
| mean_reversion | 51.2 | 56.5% | +0.058% | 2.41 | 7.10 | 50.0% |
| **routed_v23** | 60.6 | 44.6% | -0.258% | -2.26 | 5.40 | 35.0% |

## Win-rate by regime (full dataset)

| Regime | v2_all | trend_follower | mean_reversion | routed_v23 |
|--------|--------|----------------|----------------|------------|
| ACCUMULATION | 46.2% | 44.4% | 47.2% | 46.2% |
| BEAR_RANGING | 46.2% | 49.7% | 45.5% | 45.5% |
| BEAR_TRENDING | 49.6% | 44.7% | 53.1% | 44.7% |
| BULL_RANGING | 47.7% | 45.5% | 48.2% | 48.2% |
| BULL_TRENDING | 50.3% | 47.8% | 52.4% | 47.8% |
| DISTRIBUTION | 46.8% | 43.3% | 48.8% | 46.8% |

## Fold-by-fold Sharpe — routed_v23

| Fold | Start | End | Signals | Win% | Sharpe |
|------|-------|-----|---------|------|--------|
| 1 | 2022-03-01 | 2022-03-08 | 83 | 42.2% | -4.52 |
| 2 | 2022-05-18 | 2022-05-25 | 24 | 20.8% | -6.60 |
| 3 | 2022-08-04 | 2022-08-11 | 96 | 59.4% | 3.01 |
| 4 | 2022-10-21 | 2022-10-28 | 133 | 42.1% | -6.15 |
| 5 | 2023-01-07 | 2023-01-14 | 130 | 42.3% | -0.65 |
| 6 | 2023-03-26 | 2023-04-02 | 32 | 56.2% | 0.33 |
| 7 | 2023-06-12 | 2023-06-19 | 30 | 40.0% | 3.37 |
| 8 | 2023-08-29 | 2023-09-05 | 57 | 22.8% | -10.64 |
| 9 | 2023-11-15 | 2023-11-22 | 17 | 29.4% | -7.00 |
| 10 | 2024-02-01 | 2024-02-08 | 79 | 49.4% | -0.15 |
| 11 | 2024-04-19 | 2024-04-26 | 72 | 43.1% | -8.47 |
| 12 | 2024-07-06 | 2024-07-13 | 23 | 34.8% | -7.62 |
| 13 | 2024-09-22 | 2024-09-29 | 80 | 57.5% | 7.06 |
| 14 | 2024-12-09 | 2024-12-16 | 25 | 32.0% | -8.29 |
| 15 | 2025-02-25 | 2025-03-04 | 15 | 53.3% | -3.49 |
| 16 | 2025-05-14 | 2025-05-21 | 20 | 65.0% | 1.77 |
| 17 | 2025-07-31 | 2025-08-07 | 50 | 60.0% | 9.12 |
| 18 | 2025-10-17 | 2025-10-24 | 50 | 52.0% | -1.66 |
| 19 | 2026-01-03 | 2026-01-10 | 118 | 55.1% | 2.20 |
| 20 | 2026-03-22 | 2026-03-29 | 77 | 35.1% | -6.77 |

## Gate checks

- ❌ routed_v23 Sharpe > v2_all
- ❌ routed_v23 win rate > 55%
- ❌ routed_v23 positive folds > 60%

## Findings & decision

**Key insight**: `mean_reversion` (rsi + bollinger) dominates across all regimes including trending ones.
In BULL_TRENDING, mean_reversion WR is **52.4%** vs trend_follower's 47.8%.
In BEAR_TRENDING, mean_reversion WR is **53.1%** vs trend_follower's 44.7%.

Routing trending-regime traffic to trend_follower therefore hurts the composite.
The only regime where trend_follower wins is BEAR_RANGING (49.7% vs 45.5%), but the sample
is small (1080 rows) and the improvement is within noise.

**Decision**: Infrastructure (StrategyV2 `detectors` param, presets, SignalEngine `regime_routing` flag)
is fully implemented but `regime_routing=False` in `loop.py` pending a better-validated detector split.
Next step: train regime-specific LightGBM models to identify which detector features are predictive
per regime before hard-coding a detector subset routing.

**macd low signal count**: Only 270 macd signals across 4 years (vs 7853 bollinger) — macd is
firing rarely, making the trend_follower preset statistically weak in practice.
