# Smart-Trader Roadmap

> 本文档记录从 POC 到生产级 AI 原生交易系统的演进路径。
> 最后更新：2026-04-17
>
> **Jira 式分解（Epic / Story / Task）**：[`docs/implementation_todos.md`](implementation_todos.md)。

---

## 近况快照（2026-04-17）

| 领域 | 状态 | 说明 |
|------|------|------|
| **规则引擎** | 生产默认 | `trader --mode rule`；趋势 / 波动率 / 信号 / 风控 / 执行链路稳定运行。 |
| **RL 实验线** | **v9 搜索终止** | v9 架构（last-token Transformer + Beta dist + EMA obs norm + FR obs）；r1–r5 全部完成 WF 评估，均未过 shadow 门禁。**r4 为 v9 最终最佳**（Sharpe -1.51，MaxDD 1.89%，Win 50%，Mean Return -0.22%）。根本瓶颈：`lookback=1` + 63K 参数在 conservative 摩擦下信息量不足，高频小仓位无法覆盖摩擦成本。 |
| **OOS 结论（v9 数据）** | **已结案** | r4 checkpoint：`checkpoints/v9_conservative_r4/best_agent.pt`（best eval -0.1089，step 33,280）。WF Mean Sharpe -1.51（未过 shadow 门禁 Sharpe > 0）。下一步：Phase 1.1 LightGBM 信号过滤器，或设计 v10 架构（lookback=10+，更大参数量）。 |
| **数据** | 完备 | FR 100%（4700 行/symbol，2022–2026）；OI 30 天窗口；1m/1h/4h 蜡烛完整（Binance，2022-01-01）。 |
| **EPIC-RL-OPT** | **已完成** | ST-OPT-01/02/03/04/05 全部落地（v9 架构优化 + reward shaping r3-r5）。见 `implementation_todos.md`。 |
| **ML 信号过滤（Phase 1.1）** | **🟡 已启动** | `collect_signal_labels.py` 已写（历史信号回放 + 标注 → Parquet）。下一步：运行采集 → LightGBM 训练 → WF 验证。 |
| **工程** | 并行推进 | Prometheus/Grafana 等在 Phase 4；Rust 执行引擎为激活目标。 |

---

## 迭代进度与冻结约定（相对 `implementation_todos.md`）

> 与 [`docs/implementation_todos.md`](implementation_todos.md) 中 **EPIC-OOS / EPIC-RL** 同步。此处为**单一事实来源**：完成后在此更新；详细 Task 勾选仍在 `implementation_todos.md`。

### Epic `EPIC-OOS` — OOS 协议与生产角色

| Story | 状态 | 说明 |
|------|------|------|
| **ST-OOS-01** 生产与实验角色 | **已完成** | 见下「运行角色约定」；`configs/README.md` 已链到本节。 |
| **ST-OOS-02** RL WF 评估协议（书面） | **部分完成** | 下表为冻结默认值；**T-OOS-02-5**（第二人同命令复现）仍待验收。 |
| **ST-OOS-03** 基线与数据就绪 | **部分完成** | **T-OOS-03-2**：开发环境已对 `ETH/USDT` + `binance` 执行 `load_data`，1m/1h/4h 均有数据（2026-04-12）。**T-OOS-03-1** 基线路径见下表，**待登记**。 |

#### 运行角色约定（ST-OOS-01）

- **生产默认下单**：`trader --mode rule`（规则引擎：趋势 / 波动 / 信号 / 风控 / 执行）。
- **RL 智能体**：仅用于 **`trader --mode rl`** 在非生产或已授权环境，或 **`shadow` / `InferenceConfig.shadow_mode`** 与规则对照；**禁止**在未书面变更本节前将 RL 设为生产默认执行路径。

#### RL walk-forward 默认协议（ST-OOS-02，书面冻结）

| 项 | 冻结默认值 | 备注 |
|----|------------|------|
| 主推进品种 | `ETH/USDT` | 增加或改为 BTC 时须更新本表并固定一次完整 WF |
| `eval_walkforward.py` `--exchange` | `binance` | 与库内 `candles.exchange` 一致 |
| `--n-folds` | `20` | |
| `--test-days` | `7` | 若改 14 等窗口，须同步更新本表与对比实验说明 |
| `--cost-profile`（对齐实验） | `conservative` | 与 `train_rl_agent.py` 训练侧**必须同值**；见 `docs/rl_train_sim_alignment.md` |
| 门禁 | `wf_conservative_gate.py` **默认档** `shadow`（`configs/wf_gates.json` 的 `default_profile`） | 严档显式加 `--profile target` |

#### 对照基线登记（ST-OOS-03）

| 角色 | 标签 / 说明 | 路径或指针 | 状态 |
|------|-------------|------------|------|
| 规则、无 RL | *待登记* | — | [ ] |
| RL v9 r2 | v9_conservative_r2，best eval +0.8422，WF Sharpe -1.15 | `checkpoints/v9_conservative_r2/best_agent.pt` | ✅ 已记录，未过门禁 |
| **RL v9 r4（v9 最终最佳）** | best eval -0.1089（step 33,280），WF Sharpe -1.51，MaxDD 1.89%，Win 50% | `checkpoints/v9_conservative_r4/best_agent.pt` | ✅ v9 基线封档，未过门禁 |

### Epic `EPIC-RL` — 管线硬化（仅登记已书面项）

| Story | 状态 | 说明 |
|------|------|------|
| **ST-RL-02** 权重约定 | **部分完成** | **T-RL-02-1**：部署、WF 与影子推理**优先 `best_agent.pt`**，不用 `final_agent.pt` 作默认——见下文 **§3.4 强化学习策略进化**。**T-RL-02-2**（各 run 目录 README）仍由各 run 负责人补。 |
| **ST-RL-01** / **ST-RL-03** | 未在本文登记完成 | 须完成「conservative 全链长跑 + 归档 JSON」后再改此处为已完成。 |

### Epic `EPIC-RL-OPT` — RL 训练正确性修复与架构优化

> 详细 Task 见 `docs/implementation_todos.md` → `EPIC-RL-OPT`。

| Story | 状态 | 说明 |
|------|------|------|
| **ST-OPT-01** 训练正确性 Bug | **已完成** | T-OPT-01-1（GAE bootstrap）、T-OPT-01-2（`n_epochs` CLI 化）、T-OPT-01-3（risk_budget → Beta 分布）全部落地。 |
| **ST-OPT-02** 评估稳定性 | **部分完成** | T-OPT-02-1（`eval_episodes` 15 + CLI）已落地；T-OPT-02-2（滑动均值早停）可选。 |
| **ST-OPT-03** 网络架构优化 | **已完成** | T-OPT-03-1（末 token）、T-OPT-03-2（state-dep std）已落地。T-OPT-03-3：v9 烟雾测试 2026-04-16 通过（5000 steps, ETH/USDT, conservative）；全量 conservative 重训 CLI 已记录；对比 WF 图标为 optional。 |
| **ST-OPT-04** 观测空间扩展 | **已完成** | T-OPT-04-1（Running obs normalization）、T-OPT-04-2（FR backfill + coverage 100%）、T-OPT-04-3（FR 特征集成，obs_dim 70→71，microstructure_dim=1）全部落地（2026-04-16）。OI 已有 30 天数据但训练覆盖率 < 2%，暂缓至 ≥ 6 个月后接入。 |
| **ST-OPT-05** Reward Shaping | **已完成** | r3（dd_terminal=1.5）过度惩罚；r4（dd_terminal=0.3, trade_penalty=0.02）为 v9 最优；r5（trade_penalty=0.04）不改善频率且劣化 WF。超参数搜索终止，r4 封档。 |

---

## POC 回顾

### 已完成的核心模块

#### 数据与基础设施
- TimescaleDB 存储 OHLC 时序数据（hypertable + unique constraint）
- GateIOClient 通过 CCXT 拉取多周期 K 线并增量同步入库
- Docker Compose 编排 7 个服务：TimescaleDB、Redis、Prometheus、Grafana、pgAdmin、Python 引擎、Rust 引擎

#### 三层信号体系
```
1d TrendEngine  →  市场结构（BULL_TRENDING / ACCUMULATION / DISTRIBUTION ...）
4h mid-tf       →  方向过滤器（MTF alignment）
1h SignalEngine →  入场信号（EMA crossover、RSI divergence、breakout ...）
```
- ConfidenceScorer：综合 regime + momentum + volume 给信号打分
- 硬阻断规则：ACCUMULATION 禁空、DISTRIBUTION 禁多

#### 三层风险管理
| 层 | 作用 |
|----|------|
| PositionSizer | Kelly 仓位 × vol 缩放 × hard_cap_pct |
| LimitsChecker | 单仓最大敞口 5%，含浮点容差修复 |
| CircuitBreaker | 日亏超阈值熔断，每日重置 |

#### VolatilityAnalyzer
两层波动率识别（VolRegime × VolState）接入主循环：
- HIGH + SPIKE → 强制跳过信号
- HIGH vol → 延长熔断冷却（3 bars → 8 bars）、提高最低置信度（0.55 → 0.75）

#### 三级动态止损
```
固定止损 → 保本止损（到达 50% TP 距离时移至入场价）→ 跟踪止损（75% TP 距离激活，1.5×ATR trail）
```

#### 动态 R:R + Pullback Entry
- Dynamic R:R：TRENDING → 4:1，RANGING → 2.5:1，HIGH vol × 0.8，LOW vol × 1.2，区间 [1.5, 6.0]
- Pullback Entry：信号收盘后等最多 3 根 K 线，以 0.3×ATR 价格优势入场

#### 主循环优化
- 第一个 tick 启动后立即执行（无需等待下一根 K 线）
- 两次 tick 之间每 5 分钟执行 mid-cycle 出场检查（SL/TP 监控）

### POC 回测结果（1h TF，约 3 个月数据）

| Symbol | 优化前 PnL | 优化后 PnL | 胜率 | MaxDD |
|--------|-----------|-----------|------|-------|
| BTC    | 基准       | +$10~19   | 稳定  | 可控   |
| ETH    | -$35.62   | +$27.93   | 50%  | 显著改善 |
| SOL    | —         | 全部跟踪止损锁利 | 100% | 0%   |

---

## 迭代路线图

### Phase 1 — 信号质量跃升
> 目标：胜率从 50% → 60%+

#### 1.1 ML 信号过滤器
在规则引擎输出候选信号之后，加一层 LightGBM 二分类过滤器：

```
规则引擎 → 候选信号 → LightGBM 过滤器 → 最终信号
```

- **特征**：RSI、ATR rank、regime one-hot 编码、成交量偏差、近 N 根 K 线序列特征
- **标签**：未来 N 根 K 线后该信号是否盈利（walk-forward 标注，避免 lookahead bias）
- **约束**：每次 retrain 必须通过 out-of-sample 验证，防止过拟合

#### 1.2 多信号投票
要求至少 2 个独立信号源同向确认，减少噪声交易：

```python
sources = [s.source for s in signals if s.signal_type == direction]
if len(sources) < 2:
    skip()
```

#### 1.3 成交量 Profile 确认
- 入场价格与 VWAP 偏差 > 1.5%：降低置信度
- 成交量低于 20 日均量 50%：跳过信号（流动性不足）

---

### Phase 2 — 自适应进化
> 目标：策略参数随市场状态自动调整，告别人工调参

#### 2.1 在线参数学习
每周末自动对核心参数进行贝叶斯优化：

```
优化目标参数：[atr_mult, rr_ratio, pullback_frac, min_confidence]
数据窗口：滚动最近 4 周
评估指标：Sharpe × (1 - MaxDD)
输出：写入 DB（backtest_results 表），版本可追溯
```

#### 2.2 Regime-Aware 策略切换
| Regime | 模式 | 调整 |
|--------|------|------|
| BULL/BEAR_TRENDING | 激进 | 更大 R:R，更松 pullback 阈值 |
| DISTRIBUTION | 防守 | 半仓，更紧止损 |
| CRISIS vol | 离场 | 持有现金，等待 regime 恢复 |

#### 2.3 多策略组合
引入策略注册机制，按 regime 选择最适合的策略：

| 策略 | 适用场景 |
|------|---------|
| Trend Following | BULL/BEAR_TRENDING |
| Mean Reversion | RANGING + LOW vol |
| Breakout | ACCUMULATION 末期放量 |
| 当前混合策略 | 保留作 baseline 对照 |

---

### Phase 3 — AI 原生能力
> 核心差异化：感知市场叙事、融合多维数据、LLM 辅助决策

#### 3.1 新闻 / 情绪信号层

新增数据源：
- CryptoCompare News API / RSS
- Twitter/X 关键账户（Coinbase、Binance、监管机构）
- Fear & Greed Index

处理流程：
```
原始新闻 → LLM 提取（entity, sentiment, urgency）→ sentiment_score
  → 纳入 ConfidenceScorer 权重
  → major_event_flag → 触发 vol_skip（等同 SPIKE 处理）
```

#### 3.2 链上数据融合（加密货币特有 Alpha）

| 指标 | 含义 | 信号用途 |
|------|------|---------|
| Exchange Net Flow | 资金净流出交易所 | 看多偏差 |
| Miner Reserve | 矿工持仓变化 | 抛压预警 |
| Realized P&L | 散户盈亏分布 | FOMO/恐慌指标 |
| Whale Tx | 大额转账 | 波动率预警 |

链上信号作为独立置信度分量，权重可学习（Glassnode / CryptoQuant / Dune Analytics）。

#### 3.3 LLM 市场叙事理解
每日一次宏观叙事分析，输出结构化风险评估：

```python
narrative = llm.analyze(f"""
  最新价格行为: {price_summary}
  新闻摘要: {news_summary}
  链上数据: {onchain_summary}

  判断当前市场叙事、主要风险事件。
  输出 JSON: {{narrative, risk_level, time_horizon}}
""")

# risk_level HIGH → 全局调低仓位上限
```

#### 3.4 强化学习策略进化

**已实现（与代码对齐）**

- **环境**：`MarketEnv` — 多周期 K 线 + 特征引擎；奖励与约束在 `env/reward.py` 等模块可调。
- **智能体**：PPO + **Meta Controller**（Transformer 骨干，`agent/`、`scripts/train_rl_agent.py`）；动作为 **结构化 Dict**（如目标仓位、持有周期离散档、风险预算等，以 `spaces.py` 为准）。
- **训练**：支持多品种、`--per-symbol` checkpoint、`--resume`、早停与 eval 曲线；checkpoint 内保存架构字段供 `eval_walkforward.py` 自动对齐 `d_model` / `n_layers` / `lookback`。
- **样本外**：`eval_walkforward.py`（`--n-folds`、`--test-days`）输出 per-fold 与 summary JSON；可选 `render_model_comparison_charts.py` 汇总多 run 作 SVG。
- **上线前验证**：`shadow` 模式记录决策并与基准对照，不默认替代规则下单。

**进行中 / 风险**

- 分布漂移与折间方差仍大；需固定 **OOS 协议**（窗口、手续费假设、品种）再比较 run。
- `final_agent.pt` 为训练结束时刻权重；**部署与 WF 优先使用 `best_agent.pt`**（eval 最优）。

**后续（路线图）**

- 尝试 SAC / 离线 RL、或规则与 RL 的 **门控混合**（仅在高置信 regime 启用 RL）。
- 自动化：定时训练 + walk-forward 门禁 + 结果入库与告警（与 Phase 4.3 合并）。

```
环境：历史 K 线 (+ 多周期特征) + 模拟/纸面执行
Agent：PPO（Meta Controller）；SAC 等为可选方向
约束：部署前 WF OOS + 最大回撤 / 折间稳定性门槛；shadow 期对照
```

---

### Phase 4 — 工程生产化
> 持续进行，与 Phase 1-3 并行推进

#### 4.1 Rust 执行引擎激活
目前 Rust 容器为占位，激活优先场景：

| 场景 | 现状 | 目标 |
|------|------|------|
| 止损触发 | Python 每 5 分钟检查 | Rust 监控 WebSocket tick，毫秒级触发 |
| 订单簿聚合 | 不支持 | Rust 实时消费深度数据，Python 通过 gRPC 查询 |
| 执行延迟 | ~300ms | < 10ms（Python 决策 → Rust 执行） |

#### 4.2 完整可观测性
```
当前：结构化日志（structlog）

新增 Prometheus metrics：
  signal_generated_total{symbol, type, regime}
  trade_opened_total / trade_closed_total{reason}
  position_pnl_gauge{symbol}
  vol_regime_gauge{regime, state}

Grafana 仪表盘：
  实时 PnL 曲线
  信号分布热图
  vol regime 时间轴
  胜率滚动窗口

告警规则：
  DrawdownAlert（回撤 > 阈值）
  SLStreak（连续止损 N 次）
  APIError（交易所连接异常）
```

#### 4.3 Walk-Forward 自动化

**已有（手动 / CI 可接）**

- 规则与回测：`scripts/run_real_backtest.py`、`run_hybrid_backtest.py` 等（见 `python/README.md`）。
- RL OOS：`scripts/eval_walkforward.py` → JSON；多模型对比图 `scripts/render_model_comparison_charts.py`。

**待办（定时与闭环）**

```bash
# 目标：每周日 00:00 自动执行（示例）
0 0 * * 0  scripts/run_real_backtest.py --symbols BTC ETH SOL --weeks 8
  → 结果写入 DB（backtest_results 表）
  → Sharpe < 0.5 → Slack / 邮件告警
  → 参数漂移 > 阈值 → 触发 Phase 2.1 重优化

# RL 侧可并列增加：eval_walkforward + 与上一版 checkpoint 指标 diff，未过阈值则阻断「替换生产/影子模型」
```

#### 4.4 多交易所 + 多资产
```
当前：Gate.io 单交易所，3 个交易对

路线：
  → 抽象 ExchangeAdapter 接口（CCXT 已支持 100+ 交易所）
  → Binance / OKX 并行运行
  → 相关性过滤（BTC/ETH 高度相关时降低总敞口）
  → 跨交易所套利检测（价差 > 手续费 × 2）
```

---

## 竞争差异化

```
普通量化系统              →    smart-trader 目标
──────────────────────────────────────────────────
固定规则信号               →   ML 过滤 + 规则混合
单一止损策略               →   三级动态止损（保本 → 跟踪）
固定仓位                   →   Kelly × vol × regime 自适应
人工调参                   →   贝叶斯在线优化
纯技术面分析               →   技术面 + 情绪 + 链上三维融合
事后分析                   →   实时可观测性 + 自动 walk-forward
单一信号源                 →   多策略投票 + LLM 叙事理解
Python 5 分钟止损检查      →   Rust 毫秒级执行
仅规则下单                 →   RL 影子对照 + OOS 协议后再考虑混合/替代
```

**在 AI 时代真正的护城河：不是更快的规则引擎，而是能感知市场叙事、自我进化参数、融合链上信息的自适应系统。**
