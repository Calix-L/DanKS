#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

PORT="${PORT:-17862}"
SMOKE_DIR="${SMOKE_DIR:-$ROOT/runs/smoke}"
mkdir -p "$SMOKE_DIR"

"$ROOT/scripts/preflight.sh" >"$SMOKE_DIR/preflight.json"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PORT="$PORT" LOG_FILE="$SMOKE_DIR/policy.log" \
  DANKS_WORKER_ID=0 DANKS_POLICY_WORKERS=1 DANKS_REUSE_PORT=0 \
  "$ROOT/scripts/run_policy_server.sh" >"$SMOKE_DIR/policy.stdout.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >"$SMOKE_DIR/health.json" 2>/dev/null; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.25
done
if [[ "$ready" != "1" ]]; then
  echo "policy server failed to start; see $SMOKE_DIR/policy.stdout.log" >&2
  exit 2
fi

"$PYTHON" "$ROOT/scripts/check_health.py" "$SMOKE_DIR/health.json"
"$PYTHON" - "$PORT" "$SMOKE_DIR/predict.json" <<'PY'
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from DanKS.gdai_adapter.gdai_payload import ogd_label_to_plm_tile

port, output = sys.argv[1:]
payload = {
    "uid": 1,
    "chair_id": 1,
    "level_value": 2,
    "must_discard": True,
    "self_hand": [ogd_label_to_plm_tile(card) for card in ("S3", "H4", "C5")],
    "remaining_counts": {"1": 3, "2": 3, "3": 3, "4": 3},
    "players": [
        {"uid": 1, "chair_id": 1},
        {"uid": 2, "chair_id": 2},
        {"uid": 3, "chair_id": 3},
        {"uid": 4, "chair_id": 4},
    ],
    "play_history": [
        {
            "uid": 2,
            "action": "play",
            "cards": [ogd_label_to_plm_tile("D6")],
        },
        {"uid": 3, "action": "pass", "cards": []},
    ],
}
request = Request(
    f"http://127.0.0.1:{port}/api/gdai/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=30) as response:
    result = json.load(response)
Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if result.get("code") != 0:
    raise SystemExit(f"prediction failed: {result}")
action = result.get("data") or {}
if action.get("action_kind") == "PASS" or not action.get("play_cards"):
    raise SystemExit(f"illegal lead prediction: {result}")
print(json.dumps({"ok": True, "action": action}, ensure_ascii=False))
PY
echo "smoke_test_passed output=$SMOKE_DIR"
