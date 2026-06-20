#!/usr/bin/env bash
# VytalLink start — preflight, bounded-retry launch (for the intermittent Apple
# MPS startup abort), PID tracking, readiness gate (process + port + /health +
# model READY), and dashboard banner. Returns non-zero if it cannot become
# healthy after the configured attempts.
#
# Tunables (conservative committed defaults):
#   STARTUP_MAX_ATTEMPTS=3              total launch attempts
#   STARTUP_RETRY_INITIAL_SECONDS=2     backoff before attempt 2
#   STARTUP_RETRY_MAX_SECONDS=5         backoff cap (attempt 3+)
#   STARTUP_HEALTH_TIMEOUT_SECONDS=45   per-attempt wait for healthy+ready
#   MPS_STARTUP_STABILIZATION_SECONDS=0 optional settle delay after launch
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
RUN_DIR="$ROOT/run"
PID_FILE="$RUN_DIR/vytallink.pid"
LOG_OUT="$ROOT/logs/app.out"
mkdir -p "$RUN_DIR" "$ROOT/logs"

STARTUP_MAX_ATTEMPTS="${STARTUP_MAX_ATTEMPTS:-3}"
STARTUP_RETRY_INITIAL_SECONDS="${STARTUP_RETRY_INITIAL_SECONDS:-2}"
STARTUP_RETRY_MAX_SECONDS="${STARTUP_RETRY_MAX_SECONDS:-5}"
STARTUP_HEALTH_TIMEOUT_SECONDS="${STARTUP_HEALTH_TIMEOUT_SECONDS:-45}"
MPS_STARTUP_STABILIZATION_SECONDS="${MPS_STARTUP_STABILIZATION_SECONDS:-0}"
export STARTUP_MAX_ATTEMPTS   # surfaced (with STARTUP_ATTEMPT) in /health.startup

log() { printf '%s\n' "$*"; }
sdiag() { printf '[startup] %s\n' "$*"; }

if [[ ! -x "$PY" ]]; then log "ERROR: venv not found. Run scripts/setup.sh first." >&2; exit 1; fi
if ! "$PY" -c "import vytallink" >/dev/null 2>&1; then
  log "ERROR: vytallink not importable. Run scripts/setup.sh." >&2; exit 1
fi

# --- preflight: config + model file (fail fast; these never become healthy on
#     retry, so they must NOT enter the retry loop) ---------------------------
PRE="$("$PY" - <<'PYEOF' 2>/dev/null
# Prints "PORT|MODE|MODEL_OK". Exits 2 on any config/import error.
import os, sys
try:
    from vytallink.config import get_settings
    from vytallink.vision.factory import build_detector
    s = get_settings()
    mode = s.detector_mode.value
    det = build_detector(s)
    mp = (getattr(det, "model_path", "") or "")
    needs_model = mode in ("yolo", "tensorrt")
    exists = bool(mp) and os.path.exists(os.path.expanduser(mp))
    print("%d|%s|%d" % (s.port, mode, 1 if (not needs_model or exists) else 0))
except Exception as exc:
    sys.stderr.write("preflight: %s\n" % type(exc).__name__)
    sys.exit(2)
PYEOF
)"
if [[ $? -ne 0 || -z "$PRE" ]]; then
  log "ERROR: configuration invalid (see settings/.env). Not retrying." >&2; exit 2
fi
PORT="${PRE%%|*}"; REST="${PRE#*|}"; MODE="${REST%%|*}"; MODEL_OK="${REST##*|}"
BASE="http://127.0.0.1:${PORT}"
sdiag "configuration loaded (detector_mode=${MODE})"
if [[ "$MODEL_OK" != "1" ]]; then
  log "ERROR: required model file is missing for DETECTOR_MODE=${MODE}. See docs/hardware_needed.md. Not retrying." >&2
  exit 3
fi
sdiag "model file resolved"

# --- is a healthy VytalLink already running? (don't start a duplicate) -------
is_vytallink_health() { curl -fsS "$BASE/health" 2>/dev/null | grep -q '"phase"'; }
if is_vytallink_health; then
  RUNPID="$(cat "$PID_FILE" 2>/dev/null || true)"
  log "VytalLink already running and healthy${RUNPID:+ (PID $RUNPID)} on port ${PORT}. Use scripts/stop.sh first." >&2
  exit 1
fi

# --- port owned by an UNRELATED process? (never kill it) ---------------------
PORT_OWNER="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
if [[ -n "$PORT_OWNER" ]]; then
  log "ERROR: port ${PORT} is held by PID ${PORT_OWNER}, which is not a healthy VytalLink. Refusing to start (will not kill it)." >&2
  exit 4
fi

# --- stale PID file from a prior crashed run ---------------------------------
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null && ps -p "$OLD_PID" -o args= 2>/dev/null | grep -q "vytallink"; then
    log "VytalLink process $OLD_PID is alive but not healthy. Use scripts/stop.sh first." >&2; exit 1
  fi
  rm -f "$PID_FILE"  # verified stale
  sdiag "removed stale PID file"
fi

# --- launch helpers ----------------------------------------------------------
APP_PID=""
trap 'sdiag "interrupted; stopping launch"; [[ -n "$APP_PID" ]] && kill -TERM "$APP_PID" 2>/dev/null; rm -f "$PID_FILE"; exit 130' INT TERM

port_free() { [[ -z "$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null | head -1)" ]]; }

model_ready() {
  # 200 from /health AND model.state == ready.
  curl -fsS "$BASE/health" 2>/dev/null | "$PY" -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
sys.exit(0 if d.get('model',{}).get('state')=='ready' else 1)" 2>/dev/null
}

wait_ready() {
  # Returns: ready | died:<rc> | timeout
  local deadline=$(( SECONDS + STARTUP_HEALTH_TIMEOUT_SECONDS ))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      wait "$APP_PID" 2>/dev/null; echo "died:$?"; return
    fi
    if model_ready; then echo "ready"; return; fi
    sleep 0.5
  done
  echo "timeout"
}

backoff_for() {  # attempt -> seconds (initial, then capped)
  local a="$1"
  if (( a <= 1 )); then echo 0; elif (( a == 2 )); then echo "$STARTUP_RETRY_INITIAL_SECONDS"; else echo "$STARTUP_RETRY_MAX_SECONDS"; fi
}

# --- bounded retry loop ------------------------------------------------------
attempt=1
while (( attempt <= STARTUP_MAX_ATTEMPTS )); do
  if (( attempt > 1 )); then
    secs="$(backoff_for "$attempt")"
    sdiag "attempt ${attempt}/${STARTUP_MAX_ATTEMPTS}: waiting ${secs}s for resources to settle"
    sleep "$secs"
    # Confirm the previous child is dead and the port is free before relaunch
    # so retries never create duplicate workers.
    [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null && { kill -TERM "$APP_PID" 2>/dev/null; sleep 1; }
    if ! port_free; then log "ERROR: port ${PORT} did not free between attempts." >&2; rm -f "$PID_FILE"; exit 4; fi
    rm -f "$PID_FILE"
  fi

  export STARTUP_ATTEMPT="$attempt"
  sdiag "attempt ${attempt}/${STARTUP_MAX_ATTEMPTS}: launching app"
  nohup "$PY" -m vytallink.app >>"$LOG_OUT" 2>&1 &
  APP_PID=$!
  echo "$APP_PID" > "$PID_FILE"
  if (( $(printf '%.0f' "$MPS_STARTUP_STABILIZATION_SECONDS") > 0 )); then sleep "$MPS_STARTUP_STABILIZATION_SECONDS"; fi

  RESULT="$(wait_ready)"
  case "$RESULT" in
    ready)
      sdiag "attempt ${attempt}: healthy and model READY"
      log "==> VytalLink is running (PID $APP_PID)."
      log "    Local:   http://127.0.0.1:${PORT}"
      LAN_IP=""
      if hostname -I >/dev/null 2>&1; then
        LAN_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192|10|172)\.' | head -1 || true)"
      elif command -v ipconfig >/dev/null 2>&1; then
        for iface in en0 en1 en2; do ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"; [[ -n "$ip" ]] && { LAN_IP="$ip"; break; }; done
      fi
      [[ -n "$LAN_IP" ]] && log "    Network: http://${LAN_IP}:${PORT}  (same Wi-Fi/LAN)"
      log "    Logs:    $LOG_OUT  and  logs/vytallink.log"
      log "    Stop:    scripts/stop.sh"
      exit 0
      ;;
    died:*)
      rc="${RESULT#died:}"
      # Abort trap (signal 6 -> rc 134) is the known transient MPS startup abort.
      if [[ "$rc" == "134" ]]; then
        sdiag "attempt ${attempt}: FAILED — process aborted (rc=${rc}, transient MPS startup abort); retry allowed"
      else
        sdiag "attempt ${attempt}: FAILED — process exited before healthy (rc=${rc}); retry allowed"
        sdiag "last log lines:"; tail -n 8 "$LOG_OUT" | sed -E 's#192\.168\.[0-9.]+#<ip>#g; s#rtsp://[^ ]*#rtsp://<redacted>#g' >&2
      fi
      ;;
    timeout)
      sdiag "attempt ${attempt}: FAILED — not healthy within ${STARTUP_HEALTH_TIMEOUT_SECONDS}s; stopping child and retrying"
      kill -TERM "$APP_PID" 2>/dev/null || true
      for _ in $(seq 1 20); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 0.3; done
      ;;
  esac
  rm -f "$PID_FILE"
  attempt=$(( attempt + 1 ))
done

log "ERROR: VytalLink did not become healthy after ${STARTUP_MAX_ATTEMPTS} attempt(s). See $LOG_OUT" >&2
exit 1
