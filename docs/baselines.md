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
| 标签 | `v9_conservative_r4_20260417` |
| 架构版本 | v9 (PPO, lookback=1, 63K params) |
| 训练步数 | 84,480（早停，patience=20） |
| Checkpoint (best) | `python/checkpoints/v9_conservative_r4/best_agent.pt` |
| Checkpoint (final) | `python/checkpoints/v9_conservative_r4/final_agent.pt` |
| Eval JSON | `python/checkpoints/v9_conservative_r4/best_agent.eval.json` |
| Cost profile | `conservative` |
| 训练详情 | `docs/training_runs/v9_conservative_r4_20260417.md` |

**WF 结果摘要**（20 folds × 7d，ETH/USDT，binance，conservative）：

| 指标 | 值 |
|------|----|
| Mean Sharpe | -1.51 |
| Mean Return | -0.22% |
| Win Rate | 50% |
| Mean MaxDD | 1.89% |
| 过 shadow 门禁 | ❌ |

**复现命令**（T-OOS-02-5 验证通过）：

```bash
cd python && uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/v9_conservative_r4/best_agent.pt \
  --symbols ETH/USDT --exchange binance \
  --n-folds 20 --test-days 7 \
  --cost-profile conservative \
  --output /tmp/verify_r4.eval.json

uv run python scripts/verify_wf_protocol.py /tmp/verify_r4.eval.json
```

---

## 升级流程

当有新 run 达到或超过当前最佳时：

1. 在此文件更新「RL baseline」表格
2. 在 `docs/training_runs/` 新增对应 run 笔记
3. 运行 `verify_wf_protocol.py` 确认 JSON 结构合规
4. 旧 checkpoint 路径保留在文件历史中（不删除）
