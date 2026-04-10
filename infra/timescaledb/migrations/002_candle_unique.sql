-- Add unique constraint on candles for idempotent upserts.
-- TimescaleDB requires unique constraints to include the partition column (time).
ALTER TABLE candles
    ADD CONSTRAINT uq_candle UNIQUE (symbol, exchange, timeframe, time);
