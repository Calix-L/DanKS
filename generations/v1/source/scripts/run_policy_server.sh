#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7862}"
LOG_FILE="${LOG_FILE:-$ROOT/eval_runs/policy_server.log}"
SELECTOR_DEVICE="${SELECTOR_DEVICE:-cpu}"

exec "$PYTHON" -u -m DanKS.plm_eval.server \
  --host "$HOST" \
  --port "$PORT" \
  --top-n 10 \
  --max-partitions 8 \
  --lead-max-partitions 8 \
  --follow-max-partitions 8 \
  --selector-device "$SELECTOR_DEVICE" \
  --selector-checkpoint "$CHECKPOINT" \
  --auto \
  --sample-temperature 1.0 \
  --seed 20260706 \
  --exact-best-cache-size 0 \
  --partitioner-cache-size 0 \
  --quiet-decisions \
  --rank-top-k-only \
  --fast-approx-rank \
  --disable-post-selector-constraints \
  --log-file "$LOG_FILE" \
  "$@"
