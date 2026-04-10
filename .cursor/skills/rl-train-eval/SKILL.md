---
name: rl-train-eval
description: >-
  Runs PPO training and walk-forward evaluation for the Meta Controller using
  TimescaleDB candles. Use when the user asks to train RL, resume training,
  evaluate checkpoints, multi-symbol BTC+ETH, or tune PPO/MarketEnv/reward.
---

# RL 训练与评估（smart-trader）

## 前置

- Docker：`timescaledb`（及可选 `redis`）已启动。
- `configs/envs/.env` + `configs/secrets/.env`：`DB_*` 正确；训练常用 `--exchange binance` 等与库内 `candles.exchange` 一致。
- `cd python`，用 `uv run`。

## 训练

```bash
uv run python scripts/train_rl_agent.py \
  --symbols BTC/USDT ETH/USDT \
  --exchange binance \
  --timesteps 500000 \
  --d-model 128 --n-layers 2 \
  --n-steps 2048 --batch-size 256 \
  --max-episode 4000 \
  --checkpoint-dir ./checkpoints/run_name
```

- 续训：`--resume ./checkpoints/run_name/checkpoint_N.pt`。
- 改 `MarketEnv` / `RewardEngine` / 观测维后，需重新对齐 `MetaController` 与 `SpaceConfig`。

## Walk-forward 评估

```bash
uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/run_name/final_agent.pt \
  --symbols BTC/USDT ETH/USDT \
  --exchange binance \
  --n-folds 20 --test-days 7
```

- `--d-model` / `--n-layers` / `--n-heads` 必须与训练一致。
- 多品种加载数据须单次 `asyncio.run(load_all_data(...))`（见 `eval_walkforward.py`），避免 asyncpg 事件循环冲突。

## Checkpoint

- `.pt` 含权重与可选优化器状态；用于续训、评估、`trader --mode rl --model-path`。
- 勿提交 Git；本地目录 `python/checkpoints/` 已在 `.gitignore`。
