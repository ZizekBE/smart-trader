---
name: data-backfill
description: >-
  Backfills OHLCV (and optional futures) from CEX into TimescaleDB via CCXT.
  Use when the user asks to backfill Binance/Gate data, fix missing candles,
  or prepare data for RL training.
---

# 数据回补（smart-trader）

## 命令

```bash
cd python
uv run python scripts/backfill_data.py \
  --exchange binance \
  --symbols BTC/USDT ETH/USDT \
  --timeframes 1m 1h 4h \
  --since 2024-04-01 \
  --skip-quality
```

- Gate.io / Binance 等由 `--exchange` 与 `CCXTAdapter` 支持情况决定。
- 大批量 1m 耗时长，可放后台跑。
- 性能：库侧可有 hypertable 索引与压缩迁移（`infra/timescaledb/migrations/`）。

## 与训练衔接

- 训练脚本按 `symbol` + `timeframe` 从 `candles` 读数；回补的 `exchange` 字段需与训练 `--exchange` 过滤一致。
