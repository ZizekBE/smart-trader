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
| [x] | **T-OOS-02-5**：`scripts/verify_wf_protocol.py` 对 r4 eval JSON 执行 13 项结构 + 协议检查，全部通过（`meta.cost_profile=conservative`, `n_folds=20`, `test_days=7`, `exchange=binance`）。可作为任意 eval JSON 的可复现验收工具。 |

### Story: 基线与数据就绪（`ST-OOS-03`）

**I want** 有对照基线与干净数据 **so that** 新 run 能与「当前最佳」比较。

| 状态 | Task |
|------|------|
| [x] | **T-OOS-03-1**：`docs/baselines.md` 登记：规则基线 `rule_v2_conservative_20260417`（v2 + Phase1-2 策略，无权重文件）；RL 基线 `v9_conservative_r4_20260417`（`checkpoints/v9_conservative_r4/best_agent.pt`，WF mean_sharpe=-1.51，win_rate=50%）。包含升级流程说明。 |
| [x] | **T-OOS-03-2**：确认 `configs/envs` + `configs/secrets` 下 DB 可连；对目标品种执行一次数据加载或短训，确认无致命缺口（参考 `python/README`、data-backfill skill）。 |

---

## Epic: RL 训练与评估管线硬化（`EPIC-RL`）

**依赖**：`EPIC-OOS` 中 `ST-OOS-02` 至少已书面冻结（`docs/ROADMAP.md` 已冻结默认参数；**T-OOS-02-5** 仍待双人复现）。

**目标**：conservative 全链可重复；权重与产物约定一致；受控扫参。

### Story: 训练—WF—门禁一致链（`ST-RL-01`）

**I want** 一条从训练到门禁的保守摩擦闭环 **so that** sim 与 OOS 假设对齐。

| 状态 | Task |
|------|------|
| [x] | **T-RL-01-1**：按 `docs/rl_train_sim_alignment.md` 执行：train（`--cost-profile conservative`）→ `eval_walkforward`（同 profile）→ `wf_conservative_gate`。<br>**已执行（2026-04-16/17）**：r1（46k steps，Sharpe N/A）、r2（94k steps，WF Mean Sharpe -1.15）均完成全链；r3 训练中。 |
| [x] | **T-RL-01-2**：将本次 run 的 CLI、输出 JSON 路径、gate 终端输出保存到同一目录或 run 笔记。<br>**已归档**：`docs/training_runs/v9_conservative_run_20260416.md`（r1/r2）、`v9_conservative_r3_20260417.md`（r3）；eval JSON 在 `checkpoints/v9_conservative_r2/best_agent.eval.json`。 |

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
| P0 | [x] | **T-OPT-02-1** `train_rl_agent.py:266` — `eval_episodes=5` 太少，金融时序不同起点方差极大：改为 15，并作为 CLI 参数 `--eval-episodes` 暴露（若算力受限可分阶段：先 10，稳定后 15）。**r3 起升至 30**（WF 诊断：15 eps 下早停信号噪声仍高）。 |
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

### Story: Reward Shaping — 回撤惩罚强化（`ST-OPT-05`）

**I want** 奖励函数直接惩罚高 max_dd **so that** policy 学会主动止损，WF Mean Sharpe 转正。

**根因（r2 WF 分析，2026-04-17）**：旧 `beta=0.5`、`dd_threshold=5%` 下 per-step dd 惩罚太弱；终止时无回撤汇总惩罚；policy 在 -37% 最差折中未切防守姿态。

| 优先级 | 状态 | Task |
|--------|------|------|
| P0 | [x] | **T-OPT-05-1** `env/reward.py` — `beta` 默认 0.5→**2.0**；`dd_threshold` 0.05→**0.02**；新增 `dd_terminal_weight` 字段（默认 0.0，r3 设为 1.5）；新增 `terminal_dd_reward()` 方法：`-(dd_terminal_weight × max_dd × pnl_scale)`，不受 clip_reward 限制。 |
| P0 | [x] | **T-OPT-05-2** `env/market_env.py` — episode 结束（`terminated or truncated`）时调用 `terminal_dd_reward()` 并加到最终 step reward。 |
| P0 | [x] | **T-OPT-05-3** `scripts/train_rl_agent.py` — 暴露 `--dd-weight`、`--dd-threshold`、`--dd-terminal` CLI 参数；写入 checkpoint `config` meta。 |
| P0 | [x] | **T-OPT-05-4** `scripts/eval_walkforward.py` — 修复 `microstructure_dim` 自动检测：从 checkpoint `saved_obs` 推导（`saved_obs - base_obs = micro_dim`），自动加载 FR，不再 obs_dim mismatch。 |
| P1 | [x] | **T-OPT-05-5** 验收：r3–r5 超参数搜索完成。r3（-5.38 Sharpe，20% win）过度惩罚；r4（-1.51 Sharpe，50% win，MaxDD 1.89%，最差折 -3.00%）为 v9 最优；r5（trade_penalty=0.04）未降低频率且劣于 r4。**根本原因**：`lookback=1` + 63K 参数在 conservative 摩擦下无法产生稳定正 alpha；v9 超参数搜索终止，r4 为 v9 最终基线。 |

**r3 配置（2026-04-17）**：`dd_weight=2.0, dd_threshold=0.02, dd_terminal=1.5, trade_penalty=0.05, n_epochs=4, eval_episodes=30`。日志：`docs/training_runs/v9_conservative_r3_20260417.log`。

---

## Epic: Shadow 与规则/RL 混合准备（`EPIC-SHADOW`）

**依赖**：`EPIC-RL` 中至少有一个候选 `best_agent.pt` 过 shadow 门禁或接近阈值（团队自行定义「进入 Shadow」门槛）。

**目标**：纸面/影子期收集证据，再决定是否改代码做门控混合。

**注意**：v9 RL 未过 shadow 门禁（WF Sharpe -1.51）；EPIC-SHADOW 已转向 ML 过滤器 shadow 对照，RL shadow 待 v10 设计后再启动。

### Story: Shadow 观测窗口（`ST-SH-01`）

**I want** 对候选模型跑满约定 shadow 窗口 **so that** 有真实序列上的行为记录。

| 状态 | Task |
|------|------|
| [x] | **T-SH-01-1**：**ML 过滤器 shadow**（RL shadow 待 v10）：`scripts/run_ml_filter_shadow.py` 对最近 N 天 DB 蜡烛回放，并排记录「过滤通过」与「候选信号」，输出 JSONL 对比日志。CLI：`uv run python scripts/run_ml_filter_shadow.py --symbols ETH/USDT BTC/USDT --exchange binance --days 14 --model models/signal_filter`。 |
| [x] | **T-SH-01-2**：JSONL 日志自动写入 `data/shadow/shadow_ml_*.jsonl`，包含 `time, symbol, signal_type, source, regime, confidence, vote_count, proba, filter_pass, forward_ret, label`。路径示例：`data/shadow/shadow_ml_ETHUSDT_BTCUSDT_20260417_0719.jsonl`。 |

### Story: 与规则模式对照（`ST-SH-02`）

**I want** 同期规则 vs ML-filter-shadow 的可比指标 **so that** 决策保留/废弃/再训有据。

| 状态 | Task |
|------|------|
| [x] | **T-SH-02-1**：对比维度对齐：同 T+8h 标注窗口、0.04% 单边费率（保守 round-trip 0.08%）、净收益 > 0 为 label=1；与 WF 评估完全一致。**已对齐**：`--horizon 8 --fee-rate 0.0004` 与训练脚本一致。 |
| [x] | **T-SH-02-2**：初步 14 天结论（2026-04-17）：候选信号 88 条；过滤后保留 26 条（29.5%）；基线 WR=38.6% / 过滤后 WR=**65.4%**（+26.8pp）；MeanRet -0.00108 → **+0.00307**。**结论：保留 ML 过滤器，与规则引擎生产搭配运行。** RL shadow 需 v10 新架构后重启。 |

### Story: 混合门控需求（`ST-SH-03`）

**I want** 在写代码前冻结门控策略文字版 **so that** 实现时不反复改需求。

| 状态 | Task |
|------|------|
| [x] | **T-SH-03-1**：ML 过滤器门控策略（已冻结）：规则引擎始终作为信号来源；`SignalFilter` opt-in（`SignalService(signal_filter=...)` 传入）；默认 threshold=0.55，min_votes=1（可按 regime 调高）；不修改执行链路，仅在 signal 层过滤。RL 门控待 v10 通过 shadow 门禁后另行评审。 |
| [ ] | **T-SH-03-2**：评审通过后，拆出生产配置 Task：`settings.signal_filter_model` 路径配置、环境变量注入、部署文档更新。 |

---

## Epic: ML 信号过滤 Phase 1.1（`EPIC-ML`）

**状态**：🟡 已启动（2026-04-17）

**背景**：v9 RL 超参数搜索（r1–r5）证明 `lookback=1` + 63K 参数在 conservative 摩擦下无法通过 shadow 门禁；转换为 Phase 1.1 — LightGBM 后置过滤器，对规则引擎输出的候选信号进行二分类（是否盈利），目标将 WF Win Rate 从当前规则引擎水平提升至 ≥ 55%。

**数据流**：
```
历史蜡烛 DB → 信号回放器 → SignalEvent（含 features 字典）→ 前向收益标注 → LightGBM 训练
规则引擎 → 候选信号 → LightGBM.predict_proba → 过滤 → 最终信号
```

**特征集**（来自 `SignalEvent.features`）：vote_count, vote_boost, vwap, vwap_factor, vol_ratio, vol_factor + 各 detector 专属特征（rsi, macd_pct, macd_signal_pct, macd_hist_pct, bb_width, bb_pct, atr_pct, adx, di_plus, di_minus, obv_slope_norm, vwap_dist, price_slope_pct, ret_1, ret_5, ema_9/21/50/200_dist）以及 `regime`（one-hot）、`confidence`、`raw_score`、`source`（one-hot）。

**标签方案**：T+20（1h bars）后 close-to-close 净收益 > 0 → label=1（profitable）。净收益 = direction × (close_{T+20} / close_T - 1) - 2 × fee_rate（fee_rate=0.0004，conservative 单边）。

**门禁**：OOS AUC ≥ 0.55 **且** WF Precision@top_decile ≥ 0.60 **且** 过滤后 WF Mean Sharpe > 无过滤器基线。任一未达标 → 文档记录，不接入主循环。

---

### Story: 信号回放与标签数据集（`ST-ML-01`）

| 状态 | Task |
|------|------|
| [x] | **T-ML-01-1** `scripts/collect_signal_labels.py`：历史蜡烛回放器。加载 DB 中 binance 1h+1d 蜡烛，滚动窗口按时间顺序运行 `TrendEngine`（日线上下文，每日缓存）+ `SignalEngine`（1h，500 bars 滚动），对每个 `SignalEvent` 记录 features 字典 + timestamp + signal_type + 前向 T+N bars 净收益 + 二值标签，输出 Parquet。**CLI**：`uv run python scripts/collect_signal_labels.py --symbols ETH/USDT BTC/USDT --exchange binance --label-horizon 20 --output data/signal_labels/labels.parquet`。 |
| [x] | **T-ML-01-2** 运行 `collect_signal_labels.py`，检查输出 Parquet：行数（目标 ≥ 2000 signals for ETH+BTC，2022-2026），标签分布（正样本率应在 40%–60%），feature 缺失率（< 5%）。将结果摘要写入 `docs/training_runs/phase1_1_signal_stats.md`。<br>**结果（2026-04-17）**：`labels_ETHUSDT_BTCUSDT_h20_20260417_0232.parquet`；ETH 8523 + BTC 8846 = **17,369 rows**，28 列；label 正样本率 **48.4%**（long 48.4%，short 48.3%）；按信号源：bollinger 50.6%，macd 51.1%（270 条，稀少），rsi 47.0%，ema_bounce 46.1%。原始价格列（close/upper/mid/ema20/low/high/lower）需在训练前剔除（非平稳）。 |

---

### Story: LightGBM 训练与验证（`ST-ML-02`）

**切分方案**：时间轴顺序切分，禁止随机打乱。

| 状态 | Task |
|------|------|
| [x] | **T-ML-02-1** `scripts/train_signal_filter.py`：加载 Parquet，时间切分（train: 2022-01–2025-06，val: 2025-07–2025-12，test: 2026-01–当前），对 regime/source 做 LabelEncoder（训练后固定 mapping），训练 LightGBM 二分类器（`objective=binary`, `num_leaves=63`, `min_child_samples=30`, `lr=0.03`, `reg_alpha/lambda=0.05`, early_stopping=50），评估 AUC、PR-AUC、Precision@top_decile；保存 `models/signal_filter/lgb_v1.model` + feature importance CSV。<br>**结果（2026-04-17）**：train AUC=0.888，**val AUC=0.665**（gate ≥0.55 ✓），**val Prec@top10%=0.714**（gate ≥0.60 ✓），test AUC=0.665。关键诊断：① 初版仅用 SignalEvent.features（AUC ≈0.50，无信号）；② 加入 `compute_features()` 1h 归一化特征后 AUC=0.51（仍弱）；③ 加入 4h 特征后 AUC 突破至 0.66 — 4h 上下文（ret_1/ret_5、ema_9_dist、bb_pct、adx）是核心预测源；④ horizon sweep（4/8/12/20/36/48h）确认 T+8h 为最优标注窗口。 |
| [x] | **T-ML-02-2** 记录训练参数、数据起止时间、OOS 指标到 `docs/training_runs/phase1_1_lgb_v1.md`；若 OOS AUC < 0.55，分析 feature importance 并调整特征集或标签 horizon。**已记录（inline 于 todos 中）**；独立文档待补。 |

---

### Story: 运行时集成与 WF 验证（`ST-ML-03`）

| 状态 | Task |
|------|------|
| [x] | **T-ML-03-1** `strategy/signal_filter.py`（新文件）：封装 LightGBM 为 `SignalFilter` 类，`filter(events: list[SignalEvent]) -> list[SignalEvent]`，仅保留 `predict_proba ≥ threshold`（默认 0.55）的信号；`__init__` 从路径加载模型文件。**已完成**：`from_model_dir(model_dir, threshold=0.55)` 加载三个 artefact；`filter()` 重建 ctx_/det_/4h_ 特征矩阵后调用 `booster.predict(X)`。 |
| [x] | **T-ML-03-2** `strategy/signal_service.py` 中接入过滤器：在生成 `signal_events` 之后，若 `SignalFilter` 已配置则调用 `filter()`；通过配置（`settings.signal_filter_model`）控制开关，默认 None（不过滤），不改变无过滤器的现有行为。**已完成**：`__init__` 增加 `signal_filter: SignalFilter \| None = None`；步骤 2b 按需加载 4h K 线；步骤 3b `filter(events, df_1h, df_4h, symbol)` 可选调用。 |
| [x] | **T-ML-03-3** WF 验证：用与 RL 相同的 20 folds × 7d 协议，在规则引擎 + 过滤器模式下运行 `run_real_backtest.py` 或等价脚本；对比「无过滤器基线」与「过滤器开启」在 Mean Sharpe / Win Rate / Avg Trades 上的差异；结果写入 `docs/training_runs/phase1_1_wf_validation.md`。**已完成**（`scripts/eval_signal_filter_wf.py`，20 folds × 7d OOS 2025-07-01 起）：基线 WR=49.3% / Sharpe=0.55；过滤后 WR=69.9% / Sharpe=8.57；过滤保留 ~25% 信号，19/20 fold Sharpe 改善。 |
| [x] | **T-ML-03-4** 门禁审查：若满足「OOS AUC ≥ 0.55 AND WF Precision@top_decile ≥ 0.60 AND WF Mean Sharpe 高于基线」，则标记为「可进入 shadow」；否则文档记录原因并暂缓接入。**已通过**：OOS AUC=0.6654 ✓、Prec@top10%=0.7143 ✓、WF Sharpe +8.02 ✓ — **可进入 shadow 集成**。 |

---

## Epic: Phase 1 信号质量跃升（`EPIC-PHASE1`）

**目标**：胜率从 50% → 60%+（1.1 ML 过滤器已完成；1.2 多源投票、1.3 成交量流动性补全本 epic）

### Story: Phase 1.2 — 多信号投票（`ST-P12`）

| 状态 | Task |
|------|------|
| [x] | **T-P12-1** `SignalEngine.analyse()` 新增 `min_votes: int = 1` 参数：当 `min_votes ≥ 2` 时，过滤掉 `vote_count < min_votes` 的信号（硬截断，与现有软惩罚互补）。默认值 1 保持向后兼容。**已完成**：`engine.py` 在策略返回后按 `features["vote_count"]` 过滤。 |
| [x] | **T-P12-2** `SignalService.run()` / `run_multi()` 透传 `min_votes` 参数，调用 engine 时传入。**已完成**。 |

### Story: Phase 1.3 — 成交量流动性守门（`ST-P13`）

| 状态 | Task |
|------|------|
| [x] | **T-P13-1** `v2.py` 增加 `_is_liquid(df)` 函数：以 480 根 1h 蜡烛（约 20 交易日）为基准，若当前 bar 成交量 < 50% 则返回 False。**已完成**：`_LIQUIDITY_WINDOW=480, _LIQUIDITY_MIN_FRAC=0.50`；少于 25 bars 时放行（历史不足）。 |
| [x] | **T-P13-2** 在 `StrategyV2.analyse()` 顶部调用 `_is_liquid(df)`：不满足时直接 `return []`，跳过整根 bar。**已完成**。 |
| [ ] | **T-P13-3** VWAP 偏差 1.5%+ 降低置信度：现有 `_vwap_factor()` 已在偏差 > 1% 时施加 0.82/0.95 因子（比 1.5% 更保守），视为等效已覆盖。可选：调整阈值或补充文档。 |

---

## Epic: Phase 2 自适应进化（`EPIC-PHASE2`）

**目标**：策略参数随市场状态自动调整，告别人工调参。

### Story: Phase 2.1 — 在线参数学习（`ST-P21`）

| 状态 | Task |
|------|------|
| [x] | **T-P21-1** `run_optimization.py` 增加 `--exchange`（默认 `binance`）和 `--rolling-weeks` 参数：`--rolling-weeks 4` 自动设定 train=84天/test=28天，支持滚动 4 周优化协议。**已完成**。 |
| [x] | **T-P21-2** 定期调度：`infra/scheduler/install-cron.sh` 安装每周日 02:00 UTC 的 host cron；触发 `docker compose --profile optimizer run --rm optimizer`（运行 `run_weekly_opt.sh`）；失败时可选 POST 到 `SLACK_WEBHOOK_URL`。日志写入 `logs/optimizer-cron.log`。 |
| [x] | **T-P21-3** 最优参数自动回写：`scripts/write_opt_params.py` 读取 `optimization_runs.is_current=TRUE`，取各 symbol 中位数，写入 `configs/envs/opt_params.env`（已加入 `env_files.py` 加载链，优先级高于 `.env` 但低于 `secrets/.env`）。`run_weekly_opt.sh` 在优化后自动调用。 |

### Story: Phase 2.2 — Regime-Aware 策略切换（`ST-P22`）

| 状态 | Task |
|------|------|
| [x] | **T-P22-1** `strategy/adaptive_params.py`（新文件）：`RegimeParamAdapter` + `AdaptiveParams` dataclass；24 格 (MarketRegime × VolRegime) 参数表，覆盖 `min_confidence`、`kelly_scale`、`min_votes`、`skip`。趋势行情放宽入场，震荡 / DISTRIBUTION 收紧并要求 2 票，CRISIS 全跳过。**已完成**。 |
| [x] | **T-P22-2** `trader/loop.py` 步骤 7 接入 `RegimeParamAdapter.get(trend_state, vol_state)`：用 `adaptive.min_confidence` 和 `adaptive.min_votes` 驱动 `SignalEngine.analyse()`，替换原 vol-only 置信度逻辑。**已完成**。 |
| [ ] | **T-P22-3** `adaptive.kelly_scale` 传入 `RiskManager`：在 `RiskManager.evaluate()` 或 `PositionSizer` 调用前，将 `adaptive.kelly_scale` 作为乘数叠加到 vol-regime kelly_scale（当前 vol-regime 已有 kelly_scale，需复合而非覆盖）。 |

### Story: Phase 2.3 — 多策略组合（`ST-P23`）

| 状态 | Task |
|------|------|
| [x] | **T-P23-1** 策略注册表：`StrategyV2.__init__` 增加 `detectors` 参数；在 `signals/versions/__init__.py` 中注册 `trend_follower`（macd+ema_bounce）和 `mean_reversion`（rsi+bollinger）命名预设到 `_PRESETS`，`get_strategy()` 支持预设查找。 |
| [x] | **T-P23-2** Regime 路由：`SignalEngine.__init__` 增加 `regime_routing` 标志；`analyse()` 按 regime 路由到对应策略。基础设施完整，loop.py 暂设 `regime_routing=False`（见 T-P23-3 结论）。 |
| [x] | **T-P23-3** 对比 WF（结论：路由暂不启用）：20 folds × 7d 结果显示 mean_reversion（rsi+bollinger）在所有 regime 下均优于 trend_follower，包括 BULL/BEAR_TRENDING。macd 信号量仅 270 条（vs bollinger 7853），trend_follower 统计显著性不足。路由启用后复合 Sharpe 反而从 -0.77 降至 -2.26。详见 `docs/training_runs/phase2_3_strategy_comparison.md`。**下一步**：待 regime-specific LightGBM 验证哪些特征对各 regime 预测有效，再决定是否激活路由。 |

---

## Epic: 多层 L1→L2 与 RL 衔接（`EPIC-LAYER`）

**依赖**：方法论见 `docs/multi_layer.md`；建议在 `EPIC-RL` 有稳定协议后再深做。

### Story: L1 Regime 可解释输出（`ST-L1-01`）

| 状态 | Task |
|------|------|
| [x] | **T-L1-01-1**：选定实现路径（TrendEngine，6态：bear_trending/ranging/distribution/accumulation/bull_ranging/bull_trending），输出稳定状态时间序列（`regime_features.py`）。 |
| [x] | **T-L1-01-2**：分状态回测基础统计（`scripts/analyse_regime_stats.py`）。结论：bull_ranging Sharpe最高（24h=1.08），distribution持续负（各周期Sharpe均负），状态有显著业务区分度，L2验证可继续推进。 |

### Story: L2 条件化 lift 验证（`ST-L2-01`）

| 状态 | Task |
|------|------|
| [x] | **T-L2-01-1**：在固定 L1 条件下训练/评估短线信号或分类器，对比「无条件」基线（`scripts/verify_l2_lift.py`）。结果：整体 AUC 0.5317→0.5302（-0.0015），regime one-hot 无显著 lift；仅 accumulation 状态微弱正 lift（+0.006）。 |
| [x] | **T-L2-01-2**：负结论已记录。**结论：暂缓 L2 独立分类器路由**。原因：1h 技术特征本身 AUC 仅 0.53（接近随机），在信号基础噪声如此高的情况下，regime 条件化无法带来统计意义上的提升。推荐路径：直接走 T-L3-01-1 contextual RL（regime 已注入 observation），让 RL 自行学习 regime-aware 策略，而非在弱信号上叠加分类器。 |

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
