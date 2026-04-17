# smart-trader · Python 交易引擎

面向中心化交易所（CEX）的 **现货 / 模拟盘** 交易与回测引擎：多周期趋势与信号 → 波动率感知仓位 → 纸面或实盘下单。支持 **规则策略**、**双 sleeve 混合策略** 与 **强化学习（实验性）** 三条主线。

---

## 一、交易引擎与策略说明

系统里「引擎」指 **谁在驱动每一根 K 线收盘后的决策链路**。策略差异主要体现在 **信号来源、资金分配与持仓逻辑**。

| 引擎 | 入口 / 典型用法 | 策略要点 |
|------|-----------------|----------|
| **规则引擎（单策略）** | `uv run trader`（默认 `--mode rule`）→ `TradingLoop` | 单品种、单一路径：1d 趋势 + 4h 过滤 + 1h 信号（RSI/MACD/布林带等）→ 置信度与波动率门控 → Kelly 类仓位 → 执行层（限价回撤、移动止损等）。适合作为 **基准策略** 与生产默认。 |
| **混合引擎（双 sleeve）** | FastAPI 启动 trader 且 `mode=hybrid` 时 → `HybridLoop` | **资金分层**：约 60% 核心仓（Core，偏 4h/1d、高置信、大盈亏比）+ 约 30% 战术仓（Tactical，1h、小仓位、时间止损）+ 预留。两套逻辑并行，由 `CapitalAllocator` / `SleeveManager` 协调。参数在 `Settings` 的 `hybrid_*` 字段。 |
| **RL 引擎（实验）** | `uv run trader --mode rl --model-path /path/to.pt` → `RLTradingLoop` | 高层 **Meta Controller**（Transformer 骨干 + 分层动作：regime / 目标仓位 / 风险预算 / 持有周期）在低层 **确定性执行**（`MarketEnv` 仿真里已对齐）上推理。需自备训练好的 checkpoint。 |
| **Shadow 对比** | `uv run trader --mode shadow --model-path ...` | RL 与规则 **并行跑或对照**（具体以 `InferenceConfig.shadow_mode` 与 `rl_loop` 实现为准），用于线上影子验证而不直接替代规则下单。 |

**回测侧对应关系**：

- 单策略 walk-forward：`scripts/run_real_backtest.py` + `BacktestEngine`
- 混合双 sleeve：`scripts/run_hybrid_backtest.py` + `HybridBacktestEngine`（与 live `HybridLoop` 资金切分对齐）

---

## 二、架构与设计模式（简图）

整体是 **分层管道 + 可替换适配器**，配置用 **Pydantic Settings** 集中注入。

```
┌─────────────────────────────────────────────────────────────┐
│  CLI: core/main.py  (--mode rule | rl | shadow)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
   TradingLoop                           RLTradingLoop
   (规则单策略)                           (CCXTAdapter + FeatureStore + MetaController)
         │
         ├── 数据: CandleIngestionService → TimescaleDB
         ├── 策略: TrendEngine / VolatilityAnalyzer / SignalEngine / ConfidenceScorer
         ├── 风控: RiskManager（Kelly、限额、熔断）
         └── 执行: ExecutionEngine + OrderManager（paper / live 同路径）

HybridLoop（双 sleeve）
         ├── CoreSleeve（长周期、大预算）
         ├── TacticalSleeve（短周期、小预算、时间止损）
         └── 共享 ExecutionEngine / DB / GateIOClient
```

**常见设计模式（与代码对应）**：

| 模式 | 在本项目中的体现 |
|------|------------------|
| **策略 / 管道** | 每个 tick 固定步骤：同步 K 线 → 检查平仓 → 趋势/波动率 → 信号 → 过滤 → 风控 → 下单（见下文「信号管道」）。 |
| **适配器** | `ExchangeAdapter` + `CCXTAdapter`：统一 Binance / Gate.io / OKX 等；`create_adapter()` 为简单工厂。 |
| **仓库（Repository）** | `CandleRepository`、`TradeRepository` 等封装异步 SQLAlchemy 访问。 |
| **依赖注入（轻量）** | 循环类构造时注入 `client`、`session_factory`、`ExecutionEngine` 等，便于测试替换。 |
| **配置对象** | `Settings`（`pydantic-settings`）+ `get_settings()` 单例；环境变量来自 `configs/envs/.env` 与 `configs/secrets/.env`（后者覆盖前者）。 |
| **分层强化学习（研究路线）** | 高层策略网络输出「意图」，低层环境与执行规则落实成交与约束（与 `MarketEnv` / PPO 训练脚本一致）。 |

---

## 三、环境与依赖（摘要）

| 依赖 | 说明 |
|------|------|
| Python | ≥ 3.11 |
| uv | 包管理与运行入口 |
| PostgreSQL + TimescaleDB | 建议 Docker：`docker compose up -d timescaledb redis` |
| 交易所 | Gate.io 等；密钥放 `configs/secrets/.env`（勿提交） |

```bash
cp configs/envs/.env.example configs/envs/.env
cp configs/secrets/.env.example configs/secrets/.env
# 仅把密码、API Key 写入 configs/secrets/.env

cd python && uv sync
```

Docker 里 Compose 仍使用 `configs/envs/.env` 注入部分变量；容器挂载整个 `configs/`，运行时同样会读取 `configs/secrets/.env`。

---

## 四、启动方式

### 1. 规则引擎（默认）

```bash
cd python
uv run trader
uv run trader --symbol ETH/USDT --signal-tf 1h --trend-tf 1d --mid-tf 4h
uv run trader --once
```

等价脚本：

```bash
uv run python scripts/run_trader.py --symbol BTC/USDT --cash 10000
```

### 2. RL / Shadow

```bash
uv run trader --mode rl --model-path ./checkpoints/v3_multi/final_agent.pt
uv run trader --mode shadow --model-path ./checkpoints/xxx.pt
```

### 3. 混合引擎（HybridLoop）

通过 **FastAPI** 启动交易循环时选择 `mode=hybrid`（见 `api/server.py` 中 trader 启动逻辑）。本地直接跑混合循环可阅读 `HybridLoop` 文档字符串并在自定义脚本中 `asyncio.run` 实例化。

### 4. 回测

```bash
uv run python scripts/run_real_backtest.py
uv run python scripts/run_real_backtest.py --symbols BTC/USDT ETH/USDT
uv run python scripts/run_hybrid_backtest.py
```

### 5. RL 训练与 walk-forward 评估

**当前最佳架构（v10）**：`lookback=10`（Transformer 序列模式）+ `regime_dim=2`（MarketRegime + VolRegime 编码注入 context）。v10 conservative r1 WF mean_sharpe = **-0.06**，较 v9（-1.51）大幅改善。

```bash
# v10 标准训练（conservative，ETH/USDT）
uv run python scripts/train_rl_agent.py \
  --symbols ETH/USDT --exchange binance \
  --cost-profile conservative \
  --lookback 10 \
  --timesteps 200000 \
  --n-steps 2048 --batch-size 256 \
  --d-model 128 --n-layers 2 --n-heads 4 \
  --n-epochs 8 --eval-episodes 30 --patience 30 \
  --trade-penalty 0.02 --dd-weight 2.0 --dd-threshold 0.02 \
  --weight-decay 2.5e-5 \
  --checkpoint-dir ./checkpoints/v10_conservative_r1 \
  --seed 42

# WF 评估（20 折 × 7d，conservative）
uv run python scripts/eval_walkforward.py \
  --checkpoint ./checkpoints/v10_conservative_r1/best_agent.pt \
  --symbols ETH/USDT --exchange binance \
  --n-folds 20 --test-days 7 \
  --cost-profile conservative \
  --output ./checkpoints/v10_conservative_r1/best_agent.eval.json

# 协议验证（13 项检查，PASS 才算合规结果）
uv run python scripts/verify_wf_protocol.py \
  ./checkpoints/v10_conservative_r1/best_agent.eval.json

# 门禁检查
uv run python scripts/wf_conservative_gate.py \
  ./checkpoints/v10_conservative_r1/best_agent.eval.json
```

`--no-regime` 可禁用 regime 特征注入（回退到 v9 行为）。训练脚本默认 **`--d-model 128 --n-layers 2 --n-heads 4 --lookback 10`**；`--cost-profile conservative` 与 `eval_walkforward.py` 共用 `smart_trader.env.sim_profiles`。若出现 `state_dict` 错误，请显式传入与训练一致的 `--d-model`、`--n-layers`、`--n-heads`、`--lookback`。

批量对比多个 checkpoint（早停探测，数据只加载一次）：

```bash
uv run python scripts/eval_checkpoint_sweep.py --checkpoint-dir ./checkpoints/rl_500k \
  --symbols BTC/USDT ETH/USDT --exchange binance
# 更密：`--grid dense`（stride=25）或 `--stride 25`；评全量编号文件：`--grid full`（等同 `--stride 0`，很慢）
# 仅指定若干 update：`--pick 200 400 600 975 --no-final`（不要 final 时加 `--no-final`）
# 默认 stride=50（在编号最小/最大与 `N%stride==0` 上取样），并附带 `final_agent.pt`。
```

### 6. 数据回补

```bash
uv run python scripts/backfill_data.py --exchange binance --symbols BTC/USDT --timeframes 1m 1h 4h
```

### 7. API / Docker

```bash
# 仓库根目录
docker compose up -d api   # 需已配置 configs/envs/.env 等
```

---

## 五、规则引擎：单根信号管道（每根信号 K 线收盘）

1. 同步 K 线入库  
2. 检查止损 / 止盈 / 移动止损  
3. 1d 趋势（`TrendEngine`）  
4. 4h 方向过滤  
5. 波动率制度（是否跳过、信号源过滤）  
6. 1h 多探测器信号排序  
7. 止损后冷却 K 线  
8. 多周期方向一致性  
9. 风控（仓位上限、熔断）  
10. 限价回撤进场；若干根内未成交则撤单  
11. 组合净值记录  

---

## 六、风险参数（环境变量中常见项）

| 变量 | 默认含义 |
|------|----------|
| `MAX_POSITION_SIZE_PCT` | 单笔名义上限占组合比例 |
| `MAX_DAILY_LOSS_PCT` / `MAX_DRAWDOWN_PCT` | 日亏与回撤熔断 |
| `CONFIDENCE_THRESHOLD` | 最低信号置信度（高波动时可能更严） |

具体数值以 `Settings` 与 `.env` 为准。

---

## 七、目录结构（精简）

```
python/
├── src/smart_trader/
│   ├── core/           # settings、main 入口、env 加载
│   ├── data/           # 入库、特征、存储
│   ├── exchange/       # 适配器与工厂
│   ├── strategy/       # 趋势、波动率、信号、置信度
│   ├── risk/           # 仓位、限额、熔断
│   ├── execution/      # 下单与成交模型
│   ├── analysis/       # 回测与绩效指标
│   ├── trader/         # TradingLoop、HybridLoop、RLTradingLoop
│   ├── sleeve/         # 混合策略 Core / Tactical / 资金分配
│   └── agent/          # RL 网络、PPO、推理配置
├── scripts/            # 训练、回测、回补、测试脚本
└── tests/
```

---

## 八、进展记录与后续待办

### 已完成（2026-04）

| 模块 | 内容 |
|------|------|
| RL 稳定性 | v10 架构：`lookback=10` + `regime_dim=2`；WF mean_sharpe -1.51 → **-0.06** |
| Regime 特征 | `env/regime_features.py`：批量预计算 MarketRegime + VolRegime，支持 1h→1d 降采样回退 |
| 参数自动写回 | `scripts/write_opt_params.py`：把 Bayesian 优化最佳参数写入 `configs/envs/opt_params.env` |
| 周度优化调度 | `infra/scheduler/install-cron.sh`（Host cron）+ `docker-compose.yml optimizer service` |
| WF 协议验证 | `scripts/verify_wf_protocol.py`：13 项检查；v9 r4 / v10 r1 均 PASS |
| 基线注册 | `docs/baselines.md`：rule_v2 + v9 r4 + v10 r1（当前最佳） |
| 信号过滤 / 自适应参数 | Phase 1.1–2.3：LightGBM 信号过滤器、RegimeParamAdapter、多策略 sleeve |
| opt_params 加载顺序 | `.env` → `opt_params.env` → `secrets/.env`（后者覆盖） |

### 下一步（优先级排序）

| 优先 | 项目 |
|------|------|
| P0 | **RL 门禁突破**：v10 r2，调高训练步数（300k）或 `--trade-penalty`，目标 mean_sharpe > 0 |
| P1 | **FR/OI 微结构特征**：T-OPT-04-2/3，需先跑 `check_futures_coverage.py` 确认覆盖率 |
| P1 | **EPIC-LAYER**：L1 Regime 路由、L2 规则 lift、L3 RL 路由（依赖 RL 稳定后） |
| P2 | **训练性能**：并行多环境 rollout、`torch.compile` / MPS/CUDA |
| P2 | **多品种 RL**：BTC/USDT + ETH/USDT 联合训练，课程学习 |
| 可选 | **Rust 执行引擎**：低延迟路径（见 `RUST_ENGINE_*` 配置） |
| 可选 | **UI / API**：混合回测、RL 状态与报表统一展示 |
| 运维 | **密钥**：仅使用 `configs/secrets/.env`；历史误提交密钥需轮换并清理 Git 历史 |

---

## 九、免责声明

本软件仅供 **学习与研究**。加密货币交易存在重大亏损风险，请勿投入无法承受损失的资金。
