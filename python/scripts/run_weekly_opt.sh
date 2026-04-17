#!/usr/bin/env bash
# Weekly optimization runner — called by ofelia scheduler (T-P21-2).
#
# Steps:
#   1. Run Bayesian walk-forward optimization (rolling 4-week window)
#   2. Write best params to configs/envs/opt_params.env (T-P21-3)
#   3. On failure: POST to SLACK_WEBHOOK_URL if set
#
# Environment:
#   SYMBOLS            space-separated, default "BTC/USDT ETH/USDT"
#   OPT_EXCHANGE       default "binance"
#   OPT_ROLLING_WEEKS  default 4
#   SLACK_WEBHOOK_URL  optional; enables failure notifications

set -euo pipefail

SYMBOLS="${SYMBOLS:-BTC/USDT ETH/USDT}"
EXCHANGE="${OPT_EXCHANGE:-binance}"
WEEKS="${OPT_ROLLING_WEEKS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ')  $*"; }

notify_failure() {
    local msg="$1"
    log "ERROR: $msg"
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\":red_circle: *smart-trader optimizer failed*\n\`\`\`${msg}\`\`\`\"}" \
            || true
    fi
}

log "=== weekly optimization start ==="
log "symbols=${SYMBOLS}  exchange=${EXCHANGE}  rolling_weeks=${WEEKS}"

# shellcheck disable=SC2086
if ! uv run python "${SCRIPT_DIR}/run_optimization.py" \
        --symbols ${SYMBOLS} \
        --exchange "${EXCHANGE}" \
        --rolling-weeks "${WEEKS}" \
        --skip-sync; then
    notify_failure "run_optimization.py failed (exit $?)"
    exit 1
fi

log "=== writing opt params ==="

# shellcheck disable=SC2086
if ! uv run python "${SCRIPT_DIR}/write_opt_params.py" \
        --symbols ${SYMBOLS}; then
    notify_failure "write_opt_params.py failed (exit $?)"
    exit 1
fi

log "=== weekly optimization complete ==="
