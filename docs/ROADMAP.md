# Smart-Trader Roadmap

> 本文档记录从 POC 到生产级 AI 原生交易系统的演进路径。
> 最后更新：2026-04-13

---

## 近况快照（2026-04）

| 领域 | 状态 | 说明 |
|------|------|------|
| **规则引擎** | 生产默认 | `trader --mode rule`；趋势 / 波动率 / 信号 / 风控 / 执行链路稳定运行。 |
| **RL 实验线** | 可训可评可影子跑 | `scripts/train_rl_agent.py`（PPO）+ `MarketEnv`；观测含多周期特征与 **序列 lookback**；`scripts/eval_walkforward.py` 做 **20 折 × N 天** OOS；`trader --mode rl` / `shadow` + `InferenceConfig.shadow_mode` 做线上/纸面对照。 |
| **OOS 结论（当前数据）** | 持续迭代 | 在统一 walk-forward（如 14 天窗口）下，**`v8_seq_reg` 系列**（`trade_penalty≈0.02`、适度正则）整体优于单纯加长步数（`v8_seq_reg_long`）或偏低惩罚版本；**`v7_long`** 仍为另一套骨干规模下的对照基线。图表见仓库根 `README.md` / `docs/assets/model_oos_comparison.svg`（由 `scripts/render_model_comparison_charts.py` 从本地 `checkpoints` 评估 JSON 生成）。 |
| **数据** | 多交易所适配 | 业务侧经 `ExchangeAdapter` / `create_adapter()`；RL 训练/回测常用 Binance 库内蜡烛；`bulk_backfill` 等脚本用于缺口回补（详见 `python` 内脚本与 skill）。 |
| **ML 信号过滤（Phase 1.1）** | 未开始 | LightGBM 后置过滤器仍属规划，与当前规则引擎并行开发优先级可按业务排期。 |
| **工程** | 并行推进 | Prometheus/Grafana 等仍在路线图 Phase 4；Rust 执行引擎仍为激活目标而非默认路径。 |

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
