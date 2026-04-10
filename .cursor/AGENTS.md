# smart-trader — Cursor Agent 指引

本文件说明在本仓库中 **推荐如何分工使用 Rules / Skills / Agent**，与 `.cursor/rules/`、`.cursor/skills/` 配合。

## 推荐 Agent 角色（逻辑上）

| 场景 | 建议 |
|------|------|
| **功能开发 / 重构** | 默认 Agent；遵守 `smart-trader-core` rule；Python 改动触发 `python-style`。 |
| **RL 实验** | 先读 `rl-train-eval` skill；改 `agent/`、`env/` 时依赖 `rl-training`、`rl-env` rules；跑脚本看 `rl-scripts` rule。 |
| **数据与库** | 用 `data-backfill` skill；改 SQL 用 `infra-db` rule。 |
| **全栈 API + UI** | 后端 `python/src/smart_trader/api/` + `ui-next` rule；对齐路由与 `ui/src/lib/api.ts`。 |
| **大范围架构取舍** | 先用 **Plan** 模式定方案，再 **Agent** 实现。 |

## Rules 一览（自动或按文件匹配）

- **always**：`rules/smart-trader-core.mdc` — 分层、密钥、adapter、变更范围。
- **Python**：`python-style.mdc`
- **RL**：`rl-training.mdc`、`rl-env.mdc`、`rl-scripts.mdc`
- **DB**：`infra-db.mdc`
- **UI**：`ui-next.mdc`
- **交易循环 / 策略 / 执行 / 风控**：`rule-engine.mdc`（`trader/`、`strategy/`、`execution/`、`risk/`、`sleeve/`）
- **FastAPI**：`api-server.mdc`（`api/`，与 `api.ts` 对齐）

## Skills 一览（按描述由 Agent 选用）

- `skills/rl-train-eval` — 训练、续训、walk-forward、checkpoint。
- `skills/data-backfill` — CEX → TimescaleDB 回补。
- `skills/config-secrets` — env/secrets 拆分与加载。

## 执行习惯

- 需要联网或 DB 时：在 Cursor 中允许相应权限；用 `uv run` 而非裸 `python`，除非文档另有说明。
- 不要删除用户本地 `python/checkpoints/*.pt`，除非用户明确要求。
