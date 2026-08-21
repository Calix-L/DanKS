#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_GDAI_BIN="$ROOT/bin/gdai_linux_local"
GDAI_BIN="${GDAI_BIN:-$DEFAULT_GDAI_BIN}"
EXPECTED_GDAI_SHA256="${EXPECTED_GDAI_SHA256:-}"
PAILEMEN_OPPONENT_ID="${PAILEMEN_OPPONENT_ID:-pailemen_native_rule_ai}"
FORMAL_PAILEMEN_EVAL="${FORMAL_PAILEMEN_EVAL:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/plm_eval/gdai}"

AI_HOST="${AI_HOST:-http://127.0.0.1:7863}"
GAME_SOCKS5="${GAME_SOCKS5:-127.0.0.1:11080}"
GAME_ADDR="${GAME_ADDR:-}"
CALL_ROBOT_URL="${CALL_ROBOT_URL:-}"
CALL_ROBOT_SOCKS5="${CALL_ROBOT_SOCKS5:-$GAME_SOCKS5}"
NOW_HMS="$(date +%H%M%S)"
START_UID="${START_UID:-$((520000 + 10#$NOW_HMS % 300000))}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
ROBOT_COUNT="${ROBOT_COUNT:-2}"
AI_TIMEOUT_SEC="${AI_TIMEOUT_SEC:-3600}"
GAME_MODE="${GAME_MODE:-1}"
GAME_ENDING="${GAME_ENDING:-1}"
GAME_LEVEL_UP="${GAME_LEVEL_UP:-1}"
# DanKS-xzz evaluation defaults to the requested no-tribute/no-return-tribute rules.
GAME_TRIBUTE="${GAME_TRIBUTE:-0}"
GAME_ONE_WAY="${GAME_ONE_WAY:-0}"
STOP_AFTER_SMALL_ROUNDS="${STOP_AFTER_SMALL_ROUNDS:-0}"
GAME_ONE_WAY_BOOL="false"
[[ "$GAME_ONE_WAY" == "1" || "$GAME_ONE_WAY" == "true" ]] && GAME_ONE_WAY_BOOL="true"
RUN_TAG="${RUN_TAG:-danrl_retrieval_2v2_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "$STOP_AFTER_SMALL_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "STOP_AFTER_SMALL_ROUNDS must be a non-negative integer: $STOP_AFTER_SMALL_ROUNDS" >&2
  exit 2
fi
if [[ "$FORMAL_PAILEMEN_EVAL" == "1" && "$STOP_AFTER_SMALL_ROUNDS" != "0" ]]; then
  echo "formal PaiLeMen evaluation requires a complete 4136 upgrade-match terminal" >&2
  exit 2
fi
if [[ "$FORMAL_PAILEMEN_EVAL" == "1" \
      && ( "$GAME_MODE" != "1" || "$GAME_ENDING" != "1" || "$GAME_LEVEL_UP" != "1" \
      || "$GAME_TRIBUTE" != "0" || "$GAME_ONE_WAY_BOOL" != "false" ) ]]; then
  echo "formal PaiLeMen evaluation requires mode=1 ending=1 level_up=1 tribute=0 one_way=false" >&2
  exit 2
fi
if [[ "$FORMAL_PAILEMEN_EVAL" == "1" && "$ROBOT_COUNT" != "2" ]]; then
  echo "formal PaiLeMen evaluation requires exactly two callRobot opponents" >&2
  exit 2
fi
if [[ -z "$GAME_ADDR" || -z "$CALL_ROBOT_URL" ]]; then
  echo "GAME_ADDR and CALL_ROBOT_URL must be supplied by the authorized runtime" >&2
  exit 2
fi
if [[ "$FORMAL_PAILEMEN_EVAL" == "1" && -z "$EXPECTED_GDAI_SHA256" ]]; then
  echo "formal evaluation requires EXPECTED_GDAI_SHA256 from the authorized runtime" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

# Some deployment containers do not include ripgrep.  The script only uses
# extended-regex search, so GNU grep is an equivalent portable fallback.
if ! command -v rg >/dev/null 2>&1; then
  rg() { grep -E "$@"; }
fi

if [[ ! -x "$GDAI_BIN" ]]; then
  echo "gdai binary not executable: $GDAI_BIN" >&2
  exit 1
fi
actual_gdai_sha256="$(sha256sum "$GDAI_BIN" | awk '{print $1}')"
if [[ "$FORMAL_PAILEMEN_EVAL" == "1" && "$actual_gdai_sha256" != "$EXPECTED_GDAI_SHA256" ]]; then
  echo "gdai binary identity mismatch: actual=$actual_gdai_sha256 expected=$EXPECTED_GDAI_SHA256 path=$GDAI_BIN" >&2
  exit 2
fi

if ! curl -fsS "$AI_HOST/health" >/dev/null; then
  echo "AI service health check failed: $AI_HOST/health" >&2
  exit 2
fi

chair_from_log() {
  perl -ne 'if(/onEnterDeskRes:.*\s([1-4])\s0\s0\sfalse/){print "$1\n"}' "$1" | tail -1
}

desk_from_create_or_recover_log() {
  local log="$1"
  local desk_id
  desk_id="$(perl -ne 'print "$1\n" if /onCreateDesk msg:.*\{0 450 ([0-9]+)\}/' "$log" 2>/dev/null | tail -1)"
  if [[ -n "$desk_id" ]]; then
    echo "$desk_id"
    return 0
  fi
  perl -ne 'print "$1\n" if /accountId:.* deskId:([1-9][0-9]*)/' "$log" 2>/dev/null | tail -1
}

create_result_from_log() {
  perl -ne 'print "$1\n" if /onCreateDesk msg:.*\{\{([0-9]+) /' "$1" 2>/dev/null | tail -1
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  create_uid=$((START_UID + attempt * 10))
  join_uid=$((create_uid + 1))
  tag="${RUN_TAG}_try${attempt}"
  create_log="$LOG_DIR/${tag}_create_${create_uid}.out"
  join_log="$LOG_DIR/${tag}_join_${join_uid}.out"
  robot_log="$LOG_DIR/${tag}_callRobot.out"

  echo "attempt=$attempt create_uid=$create_uid join_uid=$join_uid"
  "$GDAI_BIN" \
    --ai_host="$AI_HOST" \
    -ai_timeout_sec="$AI_TIMEOUT_SEC" \
    -game_socks5="$GAME_SOCKS5" \
    -game_addr="$GAME_ADDR" \
    -logtostderr=true -v=1 \
    -create_desk=true \
    -game_mode="$GAME_MODE" \
    -game_ending="$GAME_ENDING" \
    -game_level_up="$GAME_LEVEL_UP" \
    -game_tribute="$GAME_TRIBUTE" \
    -game_one_way="$GAME_ONE_WAY_BOOL" \
    -count=1 \
    -start_uid="$create_uid" \
    >"$create_log" 2>&1 &
  create_pid=$!

  desk_id=""
  for _ in $(seq 1 100); do
    desk_id="$(desk_from_create_or_recover_log "$create_log")"
    [[ -n "$desk_id" ]] && break
    kill -0 "$create_pid" 2>/dev/null || break
    sleep 0.5
  done
  if [[ -z "$desk_id" ]]; then
    create_result="$(create_result_from_log "$create_log")"
    if [[ -n "$create_result" ]]; then
      echo "desk id not found; onCreateDesk result=$create_result; stopping create_pid=$create_pid log=$create_log"
    else
      echo "desk id not found; stopping create_pid=$create_pid log=$create_log"
    fi
    kill "$create_pid" 2>/dev/null || true
    continue
  fi

  for _ in $(seq 1 80); do
    grep -qE 'id:4133 data:' "$create_log" 2>/dev/null && break
    kill -0 "$create_pid" 2>/dev/null || break
    sleep 0.25
  done
  expected_one_way="$GAME_ONE_WAY_BOOL"
  if ! grep -qE 'id:4133 data:' "$create_log" 2>/dev/null \
      || ! grep -qE "\"mode\":${GAME_MODE}([,}])" "$create_log" \
      || ! grep -qE "\"ending\":${GAME_ENDING}([,}])" "$create_log" \
      || ! grep -qE "\"level_up\":${GAME_LEVEL_UP}([,}])" "$create_log" \
      || ! grep -qE "\"tribute\":${GAME_TRIBUTE}([,}])" "$create_log" \
      || ! grep -qE "\"one_way\":${expected_one_way}([,}])" "$create_log"; then
    echo "room rule mismatch; stopping create_pid=$create_pid desk_id=$desk_id log=$create_log"
    grep -E 'id:4133 data:' "$create_log" 2>/dev/null | tail -1 || true
    kill "$create_pid" 2>/dev/null || true
    continue
  fi
  echo "room_rule_ok desk_id=$desk_id mode=$GAME_MODE ending=$GAME_ENDING level_up=$GAME_LEVEL_UP tribute=$GAME_TRIBUTE one_way=$expected_one_way"

  "$GDAI_BIN" \
    --ai_host="$AI_HOST" \
    -ai_timeout_sec="$AI_TIMEOUT_SEC" \
    -game_socks5="$GAME_SOCKS5" \
    -game_addr="$GAME_ADDR" \
    -logtostderr=true -v=1 \
    -enter_desk_id="$desk_id" \
    -count=1 \
    -start_uid="$join_uid" \
    >"$join_log" 2>&1 &
  join_pid=$!

  for _ in $(seq 1 80); do
    grep -qE 'onEnterDeskRes' "$join_log" 2>/dev/null && break
    kill -0 "$join_pid" 2>/dev/null || break
    sleep 0.5
  done
  sleep 0.5

  create_chair="$(chair_from_log "$create_log")"
  join_chair="$(chair_from_log "$join_log")"
  echo "desk_id=$desk_id create_pid=$create_pid join_pid=$join_pid create_chair=$create_chair join_chair=$join_chair"
  echo "create_log=$create_log"
  echo "join_log=$join_log"

  if [[ -n "$create_chair" && -n "$join_chair" ]]; then
    diff=$((create_chair > join_chair ? create_chair - join_chair : join_chair - create_chair))
    if [[ "$diff" -eq 2 ]]; then
      echo "opposite_ok desk_id=$desk_id"
      echo "calling $ROBOT_COUNT PLM robots"
      if [[ -n "$CALL_ROBOT_SOCKS5" ]]; then
        robot_response="$(curl -fsS --socks5 "$CALL_ROBOT_SOCKS5" -d "deskId=${desk_id}&count=${ROBOT_COUNT}" "$CALL_ROBOT_URL")"
      else
        robot_response="$(curl -fsS -d "deskId=${desk_id}&count=${ROBOT_COUNT}" "$CALL_ROBOT_URL")"
      fi
      printf '%s\n' "$robot_response" | tee "$robot_log"
      if [[ "$robot_response" != "ok" ]]; then
        echo "callRobot identity gate failed: response=$robot_response desk_id=$desk_id" >&2
        kill "$create_pid" "$join_pid" 2>/dev/null || true
        wait "$create_pid" 2>/dev/null || true
        wait "$join_pid" 2>/dev/null || true
        exit 2
      fi
      if [[ "$FORMAL_PAILEMEN_EVAL" == "1" ]]; then
        echo "native_opponent_verified opponent_id=$PAILEMEN_OPPONENT_ID gdai_sha256=$actual_gdai_sha256 call_robot_url=$CALL_ROBOT_URL robot_count=$ROBOT_COUNT"
      else
        echo "call_robot_joined_unverified_protocol gdai_sha256=$actual_gdai_sha256 robot_count=$ROBOT_COUNT"
      fi
      echo
      echo "SUCCESS desk_id=$desk_id create_pid=$create_pid join_pid=$join_pid tag=$tag"
      echo "Logs: create=$create_log join=$join_log robot=$robot_log"
      while kill -0 "$create_pid" 2>/dev/null || kill -0 "$join_pid" 2>/dev/null; do
        if [[ "$STOP_AFTER_SMALL_ROUNDS" -gt 0 ]]; then
          completed_small_rounds="$(perl -ne '$n++ if /id:4127 data:/; END { print $n + 0 }' "$create_log" 2>/dev/null)"
          if [[ "$completed_small_rounds" -ge "$STOP_AFTER_SMALL_ROUNDS" ]]; then
            echo "small-round target reached ($completed_small_rounds/$STOP_AFTER_SMALL_ROUNDS); stopping clients"
            sleep 1
            kill "$create_pid" "$join_pid" 2>/dev/null || true
            wait "$create_pid" 2>/dev/null || true
            wait "$join_pid" 2>/dev/null || true
            break
          fi
        fi
        # gdai_linux_local keeps its socket open after the remote server sends
        # the final match balance. Stop only after that authoritative terminal
        # event so batch evaluation completes without truncating a match.
        if grep -qE 'id:4136 data:' "$create_log" "$join_log" 2>/dev/null; then
          echo "final balance received; stopping clients"
          sleep 1
          kill "$create_pid" "$join_pid" 2>/dev/null || true
          wait "$create_pid" 2>/dev/null || true
          wait "$join_pid" 2>/dev/null || true
          break
        fi
        sleep 2
      done
      echo "clients exited"
      exit 0
    fi
  fi

  echo "not opposite; stopping create_pid=$create_pid join_pid=$join_pid"
  kill "$create_pid" "$join_pid" 2>/dev/null || true
  sleep 1
done

echo "FAILED: no opposite seats found after $MAX_ATTEMPTS attempts" >&2
exit 1
