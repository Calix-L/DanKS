#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

MATCHES="${MATCHES:-1}"
PORT="${PORT:-7863}"
RUN_TAG="${RUN_TAG:-u400000_vs_plm_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/plm_eval/$RUN_TAG}"
GAME_SOCKS5="${GAME_SOCKS5:-127.0.0.1:11080}"
CALL_ROBOT_SOCKS5="${CALL_ROBOT_SOCKS5:-$GAME_SOCKS5}"
BASE_UID="${BASE_UID:-$((500000 + ($(date +%s) % 200000)))}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

if [[ ! "$MATCHES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MATCHES must be a positive integer: $MATCHES" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/clients" "$RUN_DIR/policy"
"$ROOT/scripts/preflight.sh" >"$RUN_DIR/preflight.json"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PORT="$PORT" LOG_FILE="$RUN_DIR/policy/server.log" \
  "$ROOT/scripts/run_policy_server.sh" \
  --trajectory-jsonl "$RUN_DIR/policy/trajectory.jsonl" \
  --trajectory-flush-every 1 \
  >"$RUN_DIR/policy/stdout.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >"$RUN_DIR/policy/health.json" 2>/dev/null; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.25
done
if [[ "$ready" != "1" ]]; then
  echo "policy server failed to start; see $RUN_DIR/policy/stdout.log" >&2
  exit 2
fi
"$PYTHON" "$ROOT/scripts/check_health.py" "$RUN_DIR/policy/health.json"

if [[ "$PREFLIGHT_ONLY" == "1" || "$PREFLIGHT_ONLY" == "true" ]]; then
  echo "preflight_only_passed run_dir=$RUN_DIR"
  exit 0
fi

for ((match = 0; match < MATCHES; match++)); do
  status="$RUN_DIR/match$(printf '%03d' "$match")_slot000.status.log"
  START_UID="$((BASE_UID + match * 1000))" \
  RUN_TAG="${RUN_TAG}_match$(printf '%03d' "$match")" \
  LOG_DIR="$RUN_DIR/clients" \
  AI_HOST="http://127.0.0.1:$PORT" \
  GAME_SOCKS5="$GAME_SOCKS5" \
  CALL_ROBOT_SOCKS5="$CALL_ROBOT_SOCKS5" \
  GDAI_BIN="$GDAI_BIN" \
  EXPECTED_GDAI_SHA256="$EXPECTED_GDAI_SHA256" \
  FORMAL_PAILEMEN_EVAL=1 \
  PAILEMEN_OPPONENT_ID=pailemen_native_rule_ai \
  GAME_MODE=1 \
  GAME_ENDING=1 \
  GAME_LEVEL_UP=1 \
  GAME_TRIBUTE=0 \
  GAME_ONE_WAY=0 \
  ROBOT_COUNT=2 \
  AI_TIMEOUT_SEC="${AI_TIMEOUT_SEC:-3600}" \
  MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}" \
  bash "$ROOT/scripts/run_two_vs_two_plm.sh" 2>&1 | tee "$status"
done

curl -fsS "http://127.0.0.1:$PORT/health" >"$RUN_DIR/policy/health.final.json"
"$PYTHON" "$ROOT/scripts/check_health.py" "$RUN_DIR/policy/health.final.json"
"$PYTHON" "$ROOT/scripts/audit_trajectory.py" "$RUN_DIR/policy/trajectory.jsonl"
"$PYTHON" -m DanKS.plm_eval.finalize \
  --run-dir "$RUN_DIR" \
  --checkpoint "$CHECKPOINT" \
  --gdai-bin "$GDAI_BIN" \
  --expected-matches "$MATCHES" \
  --output "$RUN_DIR/pailemen_eval_manifest.json"

echo "evaluation_complete run_dir=$RUN_DIR"
