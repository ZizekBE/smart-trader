---
name: config-secrets
description: >-
  Loads and splits smart-trader environment between public configs/envs and
  gitignored configs/secrets. Use when editing .env, DB credentials, API keys,
  Docker Compose env, or troubleshooting pydantic Settings validation.
---

# 配置与密钥（smart-trader）

## 文件

| 路径 | 用途 |
|------|------|
| `configs/envs/.env` | 非敏感：主机、端口、交易模式、币种列表、风控比例等 |
| `configs/secrets/.env` | 敏感：`DB_PASSWORD`、各所 API Key、`GRAFANA_PASSWORD` 等 |
| `configs/envs/.env.example` / `configs/secrets/.env.example` | 可提交的模板 |

## 加载顺序

- `smart_trader.core.env_files.load_repo_env_files()`：先 `envs/.env`（`setdefault`），再 `secrets/.env`（覆盖同名变量）。
- `get_settings()` 在构造 `Settings` 前会调用上述加载。

## 注意

- `.env` 行内注释不要用 `#` 粘在值后面（除非按项目已有解析规则处理）；有问题时把注释单独成行。
- 可选：`SMART_TRADER_ROOT` 指定仓库根（非标准工作目录时）。
