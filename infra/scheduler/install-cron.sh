#!/usr/bin/env bash
# Install a weekly host cron entry for the Bayesian optimizer (T-P21-2).
#
# Runs every Sunday at 02:00 UTC:
#   docker compose --profile optimizer run --rm optimizer
#
# Usage:
#   bash infra/scheduler/install-cron.sh          # install
#   bash infra/scheduler/install-cron.sh --remove  # remove

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="${REPO_DIR}/logs/optimizer-cron.log"
CRON_TAG="smart-trader-optimizer"

CRON_CMD="0 2 * * 0 cd ${REPO_DIR} && docker compose --profile optimizer run --rm optimizer >> ${LOG_FILE} 2>&1 # ${CRON_TAG}"

if [[ "${1:-}" == "--remove" ]]; then
    crontab -l 2>/dev/null | grep -v "${CRON_TAG}" | crontab -
    echo "Removed cron entry for ${CRON_TAG}"
    exit 0
fi

mkdir -p "${REPO_DIR}/logs"

# Add only if not already present
if crontab -l 2>/dev/null | grep -q "${CRON_TAG}"; then
    echo "Cron entry already installed (${CRON_TAG})"
else
    (crontab -l 2>/dev/null; echo "${CRON_CMD}") | crontab -
    echo "Installed cron entry:"
    echo "  ${CRON_CMD}"
fi

echo ""
echo "To run manually:"
echo "  cd ${REPO_DIR} && docker compose --profile optimizer run --rm optimizer"
echo ""
echo "To remove:"
echo "  bash infra/scheduler/install-cron.sh --remove"
