# Smart-Trader Roadmap

> 本文档记录从 POC 到生产级 AI 原生交易系统的演进路径。
> 最后更新：2026-03-26

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

```
环境：历史 K 线 + 实时模拟盘
Agent：PPO / SAC
动作空间：{signal_threshold, atr_mult, rr_ratio, hold_bars}
奖励函数：Sharpe × (1 - MaxDD) - 交易成本
约束：每次部署前必须通过 OOS 回测 + 最大回撤门槛
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
```bash
# 每周日 00:00 自动执行
0 0 * * 0  scripts/run_real_backtest.py --symbols BTC ETH SOL --weeks 8
  → 结果写入 DB（backtest_results 表）
  → Sharpe < 0.5 → Slack 告警
  → 参数漂移 > 阈值 → 触发 Phase 2.1 重优化
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
```

**在 AI 时代真正的护城河：不是更快的规则引擎，而是能感知市场叙事、自我进化参数、融合链上信息的自适应系统。**
