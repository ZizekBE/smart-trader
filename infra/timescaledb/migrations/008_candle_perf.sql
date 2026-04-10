-- ============================================================
-- Candle query performance optimisation
-- ============================================================
--
-- Problems addressed:
--   1. Planner picks candles_time_idx (time-only) and post-filters
--      symbol/timeframe → reads unnecessary rows in every chunk.
--   2. Default 7-day chunk interval creates 100+ tiny chunks for
--      2 years of 1m data → excessive merge-append overhead.
--   3. No compression on historical chunks → inflated I/O.
--
-- Changes:
--   • Add covering index on (exchange, symbol, timeframe, time ASC)
--     with INCLUDE (open, high, low, close, volume) for index-only scans.
--   • Widen chunk interval from 7 days → 30 days (fewer chunks).
--   • Enable native TimescaleDB compression on chunks older than 7 days.
-- ============================================================

-- 1. Covering index for the primary read pattern:
--    SELECT time, open, high, low, close, volume
--    FROM candles
--    WHERE exchange = ? AND symbol = ? AND timeframe = ?
--    ORDER BY time ASC
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candles_covering
    ON candles (exchange, symbol, timeframe, time ASC)
    INCLUDE (open, high, low, close, volume);

-- 2. Widen chunk interval to ~30 days (reduces chunk count ~4x).
--    Only affects future chunks; existing small chunks remain until
--    reindex/repartition is run separately.
SELECT set_chunk_time_interval('candles', INTERVAL '30 days');

-- 3. Enable native compression on the hypertable.
--    Segment by (exchange, symbol, timeframe) — each segment
--    becomes a highly compressible column store.
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'exchange, symbol, timeframe',
    timescaledb.compress_orderby = 'time ASC'
);

-- 4. Compress all chunks older than 7 days.
SELECT compress_chunk(c, if_not_compressed => true)
FROM show_chunks('candles', older_than => INTERVAL '7 days') AS c;

-- 5. Add a compression policy to auto-compress future chunks
--    once they're older than 7 days.
SELECT add_compression_policy('candles', INTERVAL '7 days', if_not_exists => true);
