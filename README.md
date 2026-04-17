# smart-trader

面向中心化交易所（CEX）的量化交易与回测工程，支持 **规则策略**、**混合双 sleeve** 与 **强化学习（实验）** 三条主线。详细使用说明见 **[python/README.md](python/README.md)**。

---

## 项目架构

```
smart-trader/
├── python/          # 核心交易引擎（策略 / RL / 回测 / 训练脚本）
├── ui/              # Next.js 前端（持仓、行情、报表）
├── infra/           # Docker Compose、TimescaleDB 迁移、调度器
├── configs/         # 环境变量与密钥（envs/ + secrets/）
└── docs/            # 路线图、基线注册、训练记录
```

三条策略主线：

| 引擎 | 说明 |
|------|------|
| **规则引擎** | 1d 趋势 + 4h 过滤 + 1h 多探测器信号 → Kelly 仓位 → 限价执行 |
| **混合双 sleeve** | Core（60%，长周期高置信）+ Tactical（30%，短周期时间止损）并行 |
| **RL 引擎（实验）** | Transformer Meta Controller，输出目标仓位 / 风险预算 / 持有周期 |

---

## 当前进展（2026-04）

### 规则策略
- Phase 1.1–2.3 已完成：LightGBM 信号过滤器、RegimeParamAdapter（趋势/震荡自适应参数）、多策略 sleeve 路由
- 每周 Bayesian 优化自动运行（Host cron → Docker），结果写入 `configs/envs/opt_params.env`

### 强化学习
- **v10 架构**：`lookback=10`（Transformer 序列模式）+ `regime_dim=2`（MarketRegime + VolRegime 注入 context）
- Walk-forward 评估（20 折 × 7d，ETH/USDT，conservative 成本）：

| 版本 | 架构 | Mean Sharpe | 备注 |
|------|------|-------------|------|
| v9 r4 | lookback=1，无 regime 上下文 | -1.51 | 已淘汰 |
| **v10 r1** | lookback=10 + regime_dim=2 | **-0.06** | 当前最佳 |

- 下一目标：mean_sharpe > 0，通过 shadow 门禁，进入影子验证阶段

### 基础设施
- WF 协议验证工具（`verify_wf_protocol.py`，13 项检查）
- 基线注册（`docs/baselines.md`）
- Walk-forward 早停 + checkpoint sweep

---

## 快速开始

```bash
# 启动基础服务
docker compose up -d timescaledb redis

# 安装依赖
cd python && uv sync

# 运行规则策略
uv run trader --symbol ETH/USDT

# RL 训练（v10）
uv run python scripts/train_rl_agent.py \
  --symbols ETH/USDT --exchange binance \
  --cost-profile conservative --lookback 10 \
  --timesteps 200000 --checkpoint-dir ./checkpoints/v10_r2

# Walk-forward 评估
uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/v10_r2/best_agent.pt \
  --symbols ETH/USDT --exchange binance \
  --n-folds 20 --test-days 7 --cost-profile conservative \
  --output ./checkpoints/v10_r2/best_agent.eval.json
```

---

## 免责声明

本软件仅供学习与研究。加密货币交易存在重大亏损风险，请勿投入无法承受损失的资金。
