# RL 训练与 walk-forward 执行成本对齐

## 背景

在默认（偏乐观）成交成本下训练、却在 **conservative** 摩擦下做 WF，会出现 **in-env eval 与 OOS WF 脱节**。训练侧现已支持 **`--cost-profile`**，与 `eval_walkforward.py` 共用 `smart_trader.env.sim_profiles.build_sim_config()`。

## 代码与默认（已落地）

| 项 | 说明 |
|----|------|
| `sim_profiles.py` | `default` / `conservative` 两套 `SimulatorConfig` |
| `train_rl_agent.py` | `MarketEnvConfig(sim_config=…)`；`--cost-profile`；`--entropy-coef` / `--entropy-coef-end`；**默认** `trade_penalty=0.02`、`weight_decay=2e-5`、`entropy_coef_end=0.004` |
| Checkpoint `config` | 训练结束写入 `cost_profile`、`train_simulator`、`trade_penalty`、`weight_decay`、`entropy_coef*` 便于追溯 |

## 推荐：ETH + conservative 一条命令（当前选定的「对齐 + 压换手 + PPO」）

以下参数为**一次性拍板**的初始配方（可在 CLI 上逐项改）：

| 参数 | 取值 | 理由 |
|------|------|------|
| `--cost-profile` | `conservative` | 与 conservative WF 同摩擦，缩小 sim 错配 |
| `--trade-penalty` | `0.035` | 在默认 0.02 上再压一档换手（上一版 0.025 WF 仍差） |
| `--patience` | `45` | 略增，避免略好即停 |
| `--max-episode` | `3200` | 略短于 4000，单 episode 方差略降（可按算力改） |
| `--weight-decay` | `2.5e-5` | 略强于脚本新默认 `2e-5`，抑制过拟合 |
| `--entropy-coef-end` | `0.004` | 与脚本默认一致；若后期仍噪大可试 `0.003` |
| `--timesteps` | `500000` | 与现有实验量级一致 |
| `--n-steps` / `--batch-size` | `2048` / `256` | 与现有 ETH 长跑一致 |

```bash
cd python && uv run python scripts/train_rl_agent.py \
  --symbols ETH/USDT \
  --exchange binance \
  --cost-profile conservative \
  --trade-penalty 0.035 \
  --patience 45 \
  --max-episode 3200 \
  --weight-decay 2.5e-5 \
  --timesteps 500000 \
  --n-steps 2048 \
  --batch-size 256 \
  --seed 42 \
  --checkpoint-dir ./checkpoints/eth_conservative_aligned_v1
```

评估与门禁（与训练 **同一** `cost-profile`）：

```bash
cd python && uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/eth_conservative_aligned_v1/best_agent.pt \
  --symbols ETH/USDT --exchange binance \
  --n-folds 20 --test-days 7 \
  --cost-profile conservative \
  --output ./checkpoints/eth_conservative_aligned_v1/wf_eth.eval.json

uv run python scripts/wf_conservative_gate.py \
  ./checkpoints/eth_conservative_aligned_v1/wf_eth.eval.json
```

## 调参顺序建议

1. 固定 **`--cost-profile conservative`**，先只动 **`--trade-penalty`**（0.03 → 0.04）。  
2. 再动 **`--max-episode`**（2800–3600）与 **`--patience`**。  
3. 最后才动 **`--weight-decay`** / **`--entropy-coef-end`**。

## 与 BTC / 多品种

BTC 若仍希望「便宜 sim 探索策略」，保持 **`--cost-profile default`**（默认）即可；仅在做 **conservative 门禁** 前对目标 checkpoint 切换训练 profile 重训一版，或接受 WF 与训练 sim 不一致的风险。
