# Baselines Registry (T-OOS-03-1)

Single source of truth for baseline checkpoint paths and labels used in
comparative evaluation.  Update this file when a new "current best" is
promoted.

---

## Rule-based baseline (no RL)

| 项目 | 值 |
|------|----|
| 标签 | `rule_v2_conservative_20260417` |
| 策略版本 | `STRATEGY_VERSION=v2` + Phase 1.1–2.3 signal filter |
| 交易模式 | `--mode rule` (TradingLoop rule path) |
| Checkpoint | 无权重文件；规则策略无需 checkpoint |
| 评估方法 | `BacktestEngine` 全量回测（`run_optimization.py` 基线路径） |
| 备注 | v2 策略已集成：LightGBM 信号过滤器、liquidity guard、RegimeParamAdapter |

**Rule baseline WF CLI（与 RL 对比时使用）**：

```bash
cd python && uv run python scripts/run_optimization.py \
  --symbols ETH/USDT BTC/USDT \
  --exchange binance \
  --days 180 \
  --trials 1 \
  --skip-sync
```

---

## RL baseline — 当前最佳

| 项目 | 值 |
|------|----|
| 标签 | `v10_conservative_r1_20260417` |
| 架构版本 | v10 (PPO, lookback=10, regime_dim=2, 63K params) |
| 训练步数 | 200,000 |
| Checkpoint (best) | `python/checkpoints/v10_conservative_r1/best_agent.pt` |
| Checkpoint (final) | `python/checkpoints/v10_conservative_r1/final_agent.pt` |
| Eval JSON | `python/checkpoints/v10_conservative_r1/best_agent.eval.json` |
| Cost profile | `conservative` |
| 训练详情 | — |

**WF 结果摘要**（20 folds × 7d，ETH/USDT，binance，conservative）：

| 指标 | 值 |
|------|----|
| Mean Sharpe | **-0.06** |
| Mean Return | -2.63% |
| Win Rate | 50% |
| Mean MaxDD | 20.57% |
| 过 shadow 门禁 | ❌（接近，需继续训练） |

**复现命令**：

```bash
cd python && uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/v10_conservative_r1/best_agent.pt \
  --symbols ETH/USDT --exchange binance \
  --n-folds 20 --test-days 7 \
  --cost-profile conservative \
  --output /tmp/verify_v10_r1.eval.json

uv run python scripts/verify_wf_protocol.py /tmp/verify_v10_r1.eval.json
```

---

## RL 历史记录

| 标签 | Mean Sharpe | 备注 |
|------|-------------|------|
| `v9_conservative_r4_20260417` | -1.51 | lookback=1，无 regime 上下文 |
| `v10_conservative_r1_20260417` | **-0.06** | lookback=10 + regime_dim=2，当前最佳 |

---

## 升级流程

当有新 run 达到或超过当前最佳时：

1. 在此文件更新「RL baseline」表格
2. 在 `docs/training_runs/` 新增对应 run 笔记
3. 运行 `verify_wf_protocol.py` 确认 JSON 结构合规
4. 旧 checkpoint 路径保留在文件历史中（不删除）
