#!/usr/bin/env bash
# VytalLink smoke test — full simulated workflow against a real running server.
# Starts the app in an isolated test configuration, exercises the API and
# dashboard, validates the fall pipeline (exactly one event + one alert,
# duplicate suppression, label, resolve), verifies persistence across a
# restart, and stops cleanly. Returns non-zero if any required check fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

PORT="${SMOKE_PORT:-5077}"
BASE="http://127.0.0.1:${PORT}"
SMOKE_DB="$ROOT/data/database/smoke_test.db"
SMOKE_LOG="$ROOT/logs/smoke_app.out"
APP_PID=""
FAILED=0

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

# --- helpers --------------------------------------------------------------
result() { printf "[%-4s] %-26s %s\n" "$1" "$2" "${3:-}"; [[ "$1" == "FAIL" ]] && FAILED=$((FAILED+1)); return 0; }
assert_eq() { if [[ "$1" == "$2" ]]; then result PASS "$3" "($1)"; else result FAIL "$3" "(got '$1' want '$2')"; fi; }
jq_py() { printf '%s' "$1" | "$PY" -c "import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print('__ERR__'); sys.exit(0)
print($2)"; }

cleanup() {
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill -TERM "$APP_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 0.3; done
    kill -KILL "$APP_PID" 2>/dev/null || true
  fi
  rm -f "$SMOKE_DB" "$SMOKE_DB"-* "$SMOKE_LOG"
}
trap cleanup EXIT

start_app() {
  # DISK_WARNING_PERCENT is raised so the smoke test validates the app pipeline
  # rather than the host's disk fill: a dev machine with a >90%-full disk would
  # otherwise report overall=degraded. Production keeps the 90% default.
  # ALERTS_ENABLED/CONSOLE_ALERTS_ENABLED are pinned ON so the alert-delivery
  # checks are hermetic and never depend on the developer's local .env (which may
  # disable alerts for a live hardware test).
  VYTALLINK_ENV=development VISION_MODE=simulation DETECTOR_MODE=simulation \
  WEARABLE_MODE=simulation VYTALLINK_PORT="$PORT" \
  VYTALLINK_DATABASE_PATH="$SMOKE_DB" VYTALLINK_LOG_DIR="$ROOT/logs" \
  WEARABLE_SAMPLE_SECONDS=1.0 DISK_WARNING_PERCENT=100.0 \
  ALERTS_ENABLED=true CONSOLE_ALERTS_ENABLED=true \
  CAMERA_1_ENABLED=false CAMERA_2_ENABLED=false \
    nohup "$PY" -m vytallink.app >>"$SMOKE_LOG" 2>&1 &
  APP_PID=$!
}

wait_health() {
  for _ in $(seq 1 40); do
    curl -fsS "$BASE/health" >/dev/null 2>&1 && return 0
    kill -0 "$APP_PID" 2>/dev/null || return 1
    sleep 0.5
  done
  return 1
}

stop_app() {
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill -TERM "$APP_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 0.3; done
    if kill -0 "$APP_PID" 2>/dev/null; then return 1; fi
  fi
  return 0
}

echo "VytalLink smoke test"
echo "============================================================"
rm -f "$SMOKE_DB" "$SMOKE_DB"-*

# --- 1. start + health ----------------------------------------------------
start_app
if wait_health; then result PASS "server_start" "(PID $APP_PID, port $PORT)"; else
  result FAIL "server_start" "did not become healthy"; tail -n 15 "$SMOKE_LOG"; echo "FAILED=$FAILED"; exit 1; fi

HEALTH="$(curl -fsS "$BASE/health")"
assert_eq "$(jq_py "$HEALTH" "d['overall']")" "ok" "health_overall"
assert_eq "$(jq_py "$HEALTH" "d['database']['status']")" "ok" "health_database"
assert_eq "$(jq_py "$HEALTH" "d['server']['running']")" "True" "health_server"

# --- 2. dashboard ---------------------------------------------------------
DASH="$(curl -fsS "$BASE/")"
if echo "$DASH" | grep -q "VytalLink"; then result PASS "dashboard" "(html served)"; else result FAIL "dashboard" "missing"; fi

# --- 3. API status --------------------------------------------------------
STATUS="$(curl -fsS "$BASE/api/status")"
assert_eq "$(jq_py "$STATUS" "d['name']")" "VytalLink" "api_status"

# --- 4. simulated vitals --------------------------------------------------
VLATEST="$(curl -fsS "$BASE/api/vitals/latest")"
HASVITAL="$(jq_py "$VLATEST" "d['vital'] is not None")"
assert_eq "$HASVITAL" "True" "simulated_vitals"

# --- 5. trigger fall: exactly one event + one alert -----------------------
curl -fsS -X POST "$BASE/api/simulation/fall" >/dev/null
EVENTS="$(curl -fsS "$BASE/api/events")"
assert_eq "$(jq_py "$EVENTS" "d['total']")" "1" "one_event"
EVT_UID="$(jq_py "$EVENTS" "d['items'][0]['event_uid']")"
assert_eq "$(jq_py "$EVENTS" "d['items'][0]['state']")" "confirmed_fall" "event_confirmed"
DETAIL="$(curl -fsS "$BASE/api/events/$EVT_UID")"
assert_eq "$(jq_py "$DETAIL" "d['alert_count']")" "1" "one_alert"
assert_eq "$(jq_py "$DETAIL" "d['alert_delivered']")" "True" "alert_delivered"

# --- 6. duplicate suppression --------------------------------------------
curl -fsS -X POST "$BASE/api/simulation/fall" >/dev/null
curl -fsS -X POST "$BASE/api/simulation/fall" >/dev/null
EVENTS2="$(curl -fsS "$BASE/api/events")"
assert_eq "$(jq_py "$EVENTS2" "d['total']")" "1" "duplicate_suppressed_event"
STATUS2="$(curl -fsS "$BASE/api/status")"
assert_eq "$(jq_py "$STATUS2" "d['counts']['alerts']")" "1" "duplicate_suppressed_alert"

# --- 7. label -------------------------------------------------------------
LBL="$(curl -fsS -X POST "$BASE/api/events/$EVT_UID/label" -H 'Content-Type: application/json' -d '{"label":"real_fall"}')"
assert_eq "$(jq_py "$LBL" "d['human_label']")" "real_fall" "label_event"

# --- 8. invalid input rejected -------------------------------------------
CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/events/$EVT_UID/label" -H 'Content-Type: application/json' -d '{"label":"nope"}')"
assert_eq "$CODE" "422" "invalid_input_rejected"

# --- 9. resolve -----------------------------------------------------------
RES="$(curl -fsS -X POST "$BASE/api/events/$EVT_UID/resolve" -H 'Content-Type: application/json' -d '{"note":"smoke test"}')"
assert_eq "$(jq_py "$RES" "d['state']")" "resolved" "resolve_event"

# --- 10. clean shutdown ---------------------------------------------------
if stop_app; then result PASS "clean_shutdown"; else result FAIL "clean_shutdown" "process did not stop"; fi
APP_PID=""

# --- 11. persistence across restart --------------------------------------
start_app
if wait_health; then
  AFTER="$(curl -fsS "$BASE/api/events/$EVT_UID")"
  assert_eq "$(jq_py "$AFTER" "d['human_label']")" "real_fall" "persist_label"
  assert_eq "$(jq_py "$AFTER" "d['state']")" "resolved" "persist_state"
  COUNT="$(curl -fsS "$BASE/api/events")"
  assert_eq "$(jq_py "$COUNT" "d['total']")" "1" "persist_event_count"
else
  result FAIL "persist_restart" "server did not restart"
fi
if stop_app; then result PASS "final_shutdown"; else result FAIL "final_shutdown" "did not stop"; fi
APP_PID=""

echo "============================================================"
if [[ "$FAILED" -eq 0 ]]; then
  echo "SMOKE TEST: PASS (all required checks passed)"
  exit 0
else
  echo "SMOKE TEST: FAIL ($FAILED check(s) failed)"
  exit 1
fi
