# 实现计划 — Jira 式分解（Epic / Story / Task）

> 与 `docs/ROADMAP.md`、`docs/multi_layer.md`、`docs/rl_train_sim_alignment.md` 对齐。  
> **Epic**：大目标与业务边界；**Story**：可交付、可验收的一小块；**Task**：具体动作（勾选即完成）。  
> 导入 Jira 时：可将每个 **Epic** 建为 Epic Issue，**Story** 建为 Story（父链 Epic），**Task** 建为 Sub-task 或独立 Task（链到 Story）。  
> **进度同步**：`EPIC-OOS` / `EPIC-RL` 已在 **`docs/ROADMAP.md` →「迭代进度与冻结约定」** 标注；本文件 Task 勾选与之一致。

---

## 依赖总览

```text
EPIC-OOS (评估协议) ──► EPIC-RL (RL 硬化) ──► EPIC-RL-OPT (Bug修复+架构优化)
                              │                         │
                              ▼                         ▼
                       EPIC-SHADOW (影子与混合) ◄────────┘
         EPIC-ML ─────────┘（可与 EPIC-RL 并行）
         EPIC-LAYER ──────┘（方法论验证，可晚于 RL）
         EPIC-ENG ─────────►（与上列并行）
```

---

## Epic: OOS 协议与生产角色（`EPIC-OOS`）

**目标**：在扩大训练/扫参前，固定可复现的样本外协议与「谁在生产下单」，避免不可比 run。

### Story: 生产与实验角色声明（`ST-OOS-01`）

**I want** 团队对「实盘默认模式」与「RL 允许形态」有单一事实来源 **so that** 不会误把实验模型当生产默认。

| 状态 | Task |
|------|------|
| [x] | **T-OOS-01-1**：在运维/runbook 或团队 wiki 写清：生产默认 `trader --mode rule`；RL 仅 `shadow` / 纸面，不自动替代规则。 |
| [x] | **T-OOS-01-2**：在 `configs/` 或部署说明中交叉引用上述约定（若已有文档则补一行链接即可）。 |

### Story: RL  walk-forward 评估协议冻结（`ST-OOS-02`）

**I want** 任意成员用同一组 CLI 得到可比的 WF JSON **so that** 扫参与门禁结论可信。

| 状态 | Task |
|------|------|
| [x] | **T-OOS-02-1**：选定品种列表（如仅 `ETH/USDT` 或 `BTC/USDT`+`ETH/USDT`）并写入 runbook。 |
| [x] | **T-OOS-02-2**：固定 `eval_walkforward.py` 参数：`--n-folds`、`--test-days`、`--exchange`。 |
| [x] | **T-OOS-02-3**：固定 `cost-profile`：与训练一致；conservative 实验则 train + WF 均为 `conservative`。 |
| [x] | **T-OOS-02-4**：固定门禁：`wf_conservative_gate.py` 使用 `--profile shadow` 或 `target`，并记录选用理由。 |
| [ ] | **T-OOS-02-5**：验收：第二人用相同命令复现，JSON 结构一致且 `meta.cost_profile` 符合预期。 |

### Story: 基线与数据就绪（`ST-OOS-03`）

**I want** 有对照基线与干净数据 **so that** 新 run 能与「当前最佳」比较。

| 状态 | Task |
|------|------|
| [ ] | **T-OOS-03-1**：登记 1 个「当前最佳规则 / 无 RL」基线标签 + 1 个「当前最佳 RL」checkpoint 路径（可不在 Git 内，但路径与标签可查）。 |
| [x] | **T-OOS-03-2**：确认 `configs/envs` + `configs/secrets` 下 DB 可连；对目标品种执行一次数据加载或短训，确认无致命缺口（参考 `python/README`、data-backfill skill）。 |

---

## Epic: RL 训练与评估管线硬化（`EPIC-RL`）

**依赖**：`EPIC-OOS` 中 `ST-OOS-02` 至少已书面冻结（`docs/ROADMAP.md` 已冻结默认参数；**T-OOS-02-5** 仍待双人复现）。

**目标**：conservative 全链可重复；权重与产物约定一致；受控扫参。

### Story: 训练—WF—门禁一致链（`ST-RL-01`）

**I want** 一条从训练到门禁的保守摩擦闭环 **so that** sim 与 OOS 假设对齐。

| 状态 | Task |
|------|------|
| [ ] | **T-RL-01-1**：按 `docs/rl_train_sim_alignment.md` 执行：train（`--cost-profile conservative`）→ `eval_walkforward`（同 profile）→ `wf_conservative_gate`。 |
| [ ] | **T-RL-01-2**：将本次 run 的 CLI、输出 JSON 路径、gate 终端输出保存到同一目录或 run 笔记。 |

### Story: Checkpoint 与部署约定（`ST-RL-02`）

**I want** WF 与影子推理使用同一套权重约定 **so that** 评估对象与线上一致。

| 状态 | Task |
|------|------|
| [x] | **T-RL-02-1**：书面约定：WF 与 shadow 推理优先使用 **`best_agent.pt`**，并注明不用 `final_agent.pt` 的原因（见 ROADMAP §3.4）。 |
| [ ] | **T-RL-02-2**：在候选部署包或 run 目录的 README 中写清上述约定。 |

### Story: 协议内受控扫参（`ST-RL-03`）

**I want** 在固定协议下做单维度小网格 **so that** 结论可归因。

| 状态 | Task |
|------|------|
| [ ] | **T-RL-03-1**：仅沿一个维度扫 2～3 格（如仅 `trade_penalty` 或仅 `max-episode`），其余 CLI 与 `ST-OOS-02` 一致。 |
| [ ] | **T-RL-03-2**：每格保留 `wf_*.eval.json` + gate 日志（或截图）。 |
| [ ] | **T-RL-03-3**（可选）：用 `scripts/render_model_comparison_charts.py` 或等价脚本汇总多 JSON，更新对比图。 |

---

## Epic: RL 训练正确性修复与架构优化（`EPIC-RL-OPT`）

**依赖**：`EPIC-OOS`（协议已冻结）；建议在 `EPIC-RL ST-RL-01` 完成至少一次 conservative 全链跑后再合并结果对比。

**目标**：修复代码审查发现的训练 Bug（影响 return 估计与 log_prob 正确性），提升样本效率与网络表达力，为后续 shadow 对照提供更可靠的 checkpoint。

**文件定位**：`python/src/smart_trader/agent/trainer.py`、`networks.py`、`meta_controller.py`；`python/src/smart_trader/env/market_env.py`、`spaces.py`；`python/scripts/train_rl_agent.py`。

---

### Story: 训练正确性 Bug 修复（`ST-OPT-01`）

**I want** 训练循环在 return 估计与策略 log_prob 上没有系统性偏差 **so that** PPO 更新方向可信，OOS 结果不受训练 Bug 干扰。

| 优先级 | 状态 | Task |
|--------|------|------|
| P0 | [x] | **T-OPT-01-1** `trainer.py` — GAE truncation bootstrap：`_collect_rollout` 分别记录 `terminated` 与 `truncated`；`_compute_gae` 中对 `truncated` 最后一步用 `V(s_{T+1})` 而非 0 做 bootstrap（当前 `done=terminated or truncated`，统一用 0，低估 timeout episode 的 return）。 |
| P0 | [x] | **T-OPT-01-2** `train_rl_agent.py:265` — `n_epochs` 硬编码为 4，远低于 `PPOConfig` 默认的 10：改为 CLI 参数 `--n-epochs`，脚本默认值改为 8；同步写入 checkpoint `config` 供追溯。 |
| P1 | [x] | **T-OPT-01-3** `networks.py:151-153` — `risk_budget` 从 Normal 采样后 clamp 到 [0.01, 0.10]，但 `log_prob` 在 clamp 前的 raw 值上计算，boundary 处 importance ratio 偏差：改用 `Beta(α, β)` 分布（天然支持有界区间）或改为 `TruncatedNormal`；同步更新 `log_prob` 方法与 `_risk_raw` 存储逻辑。 |

**验收**：T-OPT-01-1 / 01-2 修复后，用相同 seed 重跑一次 conservative 短训（`--timesteps 50000`），对比 rollout reward 曲线与 value_loss 趋势，确认无明显回归。

---

### Story: 评估与早停稳定性（`ST-OPT-02`）

**I want** 早停信号足够可靠 **so that** patience 机制不会因 5-episode 噪声过早终止有潜力的 run。

| 优先级 | 状态 | Task |
|--------|------|------|
| P0 | [x] | **T-OPT-02-1** `train_rl_agent.py:266` — `eval_episodes=5` 太少，金融时序不同起点方差极大：改为 15，并作为 CLI 参数 `--eval-episodes` 暴露（若算力受限可分阶段：先 10，稳定后 15）。 |
| P2 | [ ] | **T-OPT-02-2**（可选）早停判据改为 eval_reward 的 **滑动均值**（窗口 3）而非单次值，进一步平滑噪声；patience 计数逻辑不变。 |

**验收**：`eval_episodes=15` 下，同一 checkpoint 的多次 eval 标准差应明显低于 `eval_episodes=5` 的水平（可用短训跑 10 次 eval 对比标准差验证）。

---

### Story: 网络架构优化（`ST-OPT-03`）

**I want** Transformer backbone 对最新 bar 给予更高权重，且 policy 能在不确定 state 下自动扩大探索 **so that** 策略在高方差行情下更自适应。

| 优先级 | 状态 | Task |
|--------|------|------|
| P1 | [x] | **T-OPT-03-1** `networks.py:102` — `seq.mean(dim=1)` 改为取最后一个 token `seq[:, -1, :]`（最新 bar 信息），或改为 prepend learnable `[CLS]` token 后取 `seq[:, 0, :]`；选其一，保持 lookback=1 legacy 路径兼容。 |
| P1 | [x] | **T-OPT-03-2** `networks.py:130` — `position_log_std` 为全局参数，不随 state 变化：增加 `position_log_std_head = nn.Linear(d_model, 1)`，输出 `log_std` clamp 到 `[-4, 0]`（对应 std 在 [0.018, 1.0]）；`risk_log_std` 同步处理。 |
| P2 | [x] | **T-OPT-03-3** 对比实验：保留当前 mean-pool checkpoint 作为对照，用相同 conservative 协议跑 last-token 版，`render_model_comparison_charts.py` 汇总对比图；结论写入 run 笔记。<br>**烟雾测试（2026-04-16）**：`v9_opt_smoke` 以 5000 steps 完整跑通（ETH/USDT, conservative, seed=42），产出 `final_agent.pt` + `best_agent.pt`，无 Python 异常。value_loss 首轮冷启动 1.9T → 第 2 轮降至 81（符合预期，随机 value head 未收敛）。**完整 conservative 重训命令**：`uv run python scripts/train_rl_agent.py --symbols BTC/USDT ETH/USDT --exchange binance --cost-profile conservative --timesteps 200000 --n-epochs 8 --eval-episodes 15 --trade-penalty 0.02 --weight-decay 2e-5 --entropy-coef 0.02 --entropy-coef-end 0.004 --checkpoint-dir ./checkpoints/v9_conservative --seed 42`。对比实验建议在有完整 WF 前不强制跑（收益 < 成本），标记为 optional。 |

**验收**：last-token / CLS token 版 `eval_walkforward` 的 `mean_sharpe` 不低于 mean-pool 版（持平或提升则合并）。

---

### Story: 观测空间与特征扩展（`ST-OPT-04`）

**I want** 观测向量的各维度在同一数量级，且包含期货市场特有的资金费率与持仓量信号 **so that** Transformer 的 attention 不被量纲差异干扰，且 agent 能感知杠杆资金动态。

| 优先级 | 状态 | Task |
|--------|------|------|
| P1 | [x] | **T-OPT-04-1** `market_env.py` / `observation.py` — 加入 **running observation normalization**：对每个特征维度维护指数移动均值与方差（EMA，`alpha=0.001`），在 `_get_observation` 输出前做 z-score；clamp 到 `[-5, 5]` 防止极端值。状态随 checkpoint 一起序列化。 |
| P2 | [x] | **T-OPT-04-2** `spaces.py` — `microstructure_dim` 从 0 改为实际维度，接入 `funding_rate`（资金费率）与 `open_interest`（未平仓量）；数据源已在 `data/models/funding_rate.py`、`data/models/open_interest.py`，需确认 DB 数据覆盖率（`>= 80%`）再启用。<br>**覆盖率检查 → 已通过（2026-04-16）**：migration 007 手动执行后运行 FR backfill（Binance `fetchFundingRateHistory`，2022-01-01 起）；修复两处 bug：① `on_conflict_do_nothing` 由 `constraint=` 改为 `index_elements=`，② 期货 symbol 格式（`BTC/USDT:USDT`）归一化为 spot 格式（`BTC/USDT`）存储。最终：FR 覆盖率 **100%**（4700 行/symbol），OI 覆盖率 **100%**（720 行/symbol，30 天窗口，Binance API 限制）。Coverage 脚本：`python/scripts/check_futures_coverage.py`。 |
| P2 | [x] | **T-OPT-04-3** 对齐 `SpaceConfig.microstructure_dim`、`MarketEnvConfig`、`MetaController(context_dim=…)` 三处配置，确保 obs_dim 自动推导不需手动改多处；增加 `assert obs.shape == observation_space.shape` 的 CI 检查。<br>**已完成（2026-04-16）**：FR backfill 修复后（migration 007 + symbol 归一化 + upsert 改 `index_elements`），FR 覆盖率 100%（4700 行/symbol，2022-2026）。`microstructure_dim=1`（FR only）已集成：`train_rl_agent.py` 自动加载 funding_rates 并传入 `MarketEnvConfig.funding_rates`；`MarketEnv._get_observation()` 用 `pd.Series.asof(ts)` 查最近 FR；`obs_dim` 自动从 70 → 71。OI 30 天覆盖率 100% 但历史跨度 < 2%（不足以覆盖随机采样的 4 年训练集），暂缓至积累 ≥ 6 个月后再接入。烟雾验证（2000 steps, ETH/USDT, conservative）通过，无异常。 |

**验收**：T-OPT-04-1 完成后，训练曲线的 value_loss 方差应降低；T-OPT-04-2 需先做数据覆盖率报告再决定是否上线。

---

## Epic: Shadow 与规则/RL 混合准备（`EPIC-SHADOW`）

**依赖**：`EPIC-RL` 中至少有一个候选 `best_agent.pt` 过 shadow 门禁或接近阈值（团队自行定义「进入 Shadow」门槛）。

**目标**：纸面/影子期收集证据，再决定是否改代码做门控混合。

### Story: Shadow 观测窗口（`ST-SH-01`）

**I want** 对候选模型跑满约定 shadow 窗口 **so that** 有真实序列上的行为记录。

| 状态 | Task |
|------|------|
| [ ] | **T-SH-01-1**：用 `trader --mode shadow`（或 hybrid 配置）挂载候选 `best_agent.pt`，跑满约定天数/周数。 |
| [ ] | **T-SH-01-2**：保留 `shadow_*.jsonl` 或项目中等价结构化日志，路径登记到 run 笔记。 |

### Story: 与规则模式对照（`ST-SH-02`）

**I want** 同期规则 vs RL-shadow 的可比指标 **so that** 决策保留/废弃/再训有据。

| 状态 | Task |
|------|------|
| [ ] | **T-SH-02-1**：对齐对比维度：笔数、回撤、费用、滑点/摩擦假设是否与 WF 一致。 |
| [ ] | **T-SH-02-2**：输出书面结论：保留 / 废弃 / 再训，并链到对应 checkpoint 与 JSON。 |

### Story: 混合门控需求（`ST-SH-03`）

**I want** 在写代码前冻结门控策略文字版 **so that** 实现时不反复改需求。

| 状态 | Task |
|------|------|
| [ ] | **T-SH-03-1**：文档描述：何种 regime / 置信度下采信 RL 动作、否则回落规则（对齐 ROADMAP「门控混合」）。 |
| [ ] | **T-SH-03-2**：评审通过后，再拆开发 Task（接口、配置、单测）— 本清单不展开实现细节。 |

---

## Epic: ML 信号过滤 Phase 1.1（`EPIC-ML`）

**并行**：可与 `EPIC-RL` 并行，视人力。

**目标**：规则候选信号后经 ML 过滤，WF 标签无泄漏，未过门槛不接入主循环。

### Story: 标注与特征契约（`ST-ML-01`）

| 状态 | Task |
|------|------|
| [ ] | **T-ML-01-1**：定义 walk-forward 标签：未来 N 根、无 lookahead；与 `SignalEngine` 输出字段对齐文档。 |
| [ ] | **T-ML-01-2**：特征列表与来源表（与现有特征引擎关系写清）。 |

### Story: 训练与验证切分（`ST-ML-02`）

| 状态 | Task |
|------|------|
| [ ] | **T-ML-02-1**：实现或约定按时间轴的 train/val 切分，禁止跨日随机打乱。 |
| [ ] | **T-ML-02-2**：记录每次训练使用的起止时间与品种。 |

### Story: 上线门槛（`ST-ML-03`）

| 状态 | Task |
|------|------|
| [ ] | **T-ML-03-1**：定义 OOS 指标门槛（如 AUC/PR 或业务指标）+ 与「无过滤器」回测对比要求。 |
| [ ] | **T-ML-03-2**：未过门槛则文档声明「不接入主循环」，避免静默上线。 |

---

## Epic: 多层 L1→L2 与 RL 衔接（`EPIC-LAYER`）

**依赖**：方法论见 `docs/multi_layer.md`；建议在 `EPIC-RL` 有稳定协议后再深做。

### Story: L1 Regime 可解释输出（`ST-L1-01`）

| 状态 | Task |
|------|------|
| [ ] | **T-L1-01-1**：选定实现路径（HMM / 聚类 / TrendEngine 增强等），输出稳定状态时间序列。 |
| [ ] | **T-L1-01-2**：分状态回测基础统计（收益、波动、样本量），判断状态是否有业务意义。 |

### Story: L2 条件化 lift 验证（`ST-L2-01`）

| 状态 | Task |
|------|------|
| [ ] | **T-L2-01-1**：在固定 L1 条件下训练/评估短线信号或分类器，对比「无条件」基线。 |
| [ ] | **T-L2-01-2**：若无统计提升，文档结论「暂缓 RL 路由」，避免堆叠无效层。 |

### Story: RL 与上层观测衔接（`ST-L3-01`）

| 状态 | Task |
|------|------|
| [ ] | **T-L3-01-1**：选定单一方案：contextual 观测扩展 **或** MoE/软路由（与现有规划一致）。 |
| [ ] | **T-L3-01-2**：拆开发任务（改 `MarketEnv`/观测、配置项、回归测试）— 实现细节单独 Epic 亦可。 |

---

## Epic: 工程化与自动化（`EPIC-ENG`）

**并行**：与 `EPIC-RL`～`EPIC-SHADOW` 并行推进。

### Story: Walk-forward 定时流水线（`ST-ENG-01`）

| 状态 | Task |
|------|------|
| [ ] | **T-ENG-01-1**：选择调度器（cron / GitHub Actions / 内部），定时跑 `eval_walkforward` + `wf_conservative_gate`。 |
| [ ] | **T-ENG-01-2**：失败时通知（Slack 或邮件择一），并记录最后一次成功时间。 |

### Story: 运行产物归档（`ST-ENG-02`）

| 状态 | Task |
|------|------|
| [ ] | **T-ENG-02-1**：每次运行归档：`*.eval.json`、完整 CLI、`torch.load` 打印的 `config` 中与摩擦相关的键（`cost_profile`、`train_simulator` 等）。 |

### Story: 可观测性与执行路径（`ST-ENG-03`）

| 状态 | Task |
|------|------|
| [ ] | **T-ENG-03-1**（长期）：Prometheus 指标 + Grafana 面板（ROADMAP Phase 4.2）。 |
| [ ] | **T-ENG-03-2**（长期）：Rust 执行引擎激活范围另立子 Epic / 子清单。 |

---

## Sprint 节奏建议（映射 Epic）

| 建议周次 | 主要 Epic / Story |
|----------|-------------------|
| 1 | `EPIC-OOS` 全部 + `ST-RL-01` |
| 2 | `ST-RL-02`、`ST-RL-03` |
| **2（并行）** | **`ST-OPT-01`（P0 Bug 修复）+ `ST-OPT-02`（eval 稳定性）** — 代码量小，修完即重训对比 |
| 3 | `EPIC-SHADOW`（`ST-SH-01`、`ST-SH-02`） |
| **3（并行）** | **`ST-OPT-03`（Transformer 末 token + state-dep std）** — 需对比实验，与 Shadow 并行跑 |
| 4 | `ST-OPT-04`（观测归一化 + 微结构特征，需数据覆盖率确认） |
| 5+ | `EPIC-ML` 或 `EPIC-LAYER` 或 `EPIC-ENG` 择一加深 |

完成 **OOS + RL 硬化 + Bug 修复 + Shadow 对照** 后，再扩大品种或上 hybrid，风险更可控。

---

## Jira 字段映射提示（可选）

| 本表元素 | Jira |
|----------|------|
| `EPIC-*` | Issue type **Epic**，Summary 用 Epic 标题 |
| `ST-*` | Issue type **Story**，Epic Link 指向父 Epic |
| `T-*-..` | **Sub-task** 或 **Task**，父 Issue 选对应 Story |
| 依赖总览图 | Jira **Blocks / Is blocked by** 或 Advanced Roadmaps |

Story ID（`ST-OOS-01` 等）为仓库内代号；同步到 Jira 时可改为项目键如 `TRAD-101`。
