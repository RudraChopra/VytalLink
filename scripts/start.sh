#!/usr/bin/env bash
# VytalLink start — validates setup, prevents duplicate launch, starts the app,
# records the PID, waits for health, and prints dashboard addresses.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
RUN_DIR="$ROOT/run"
PID_FILE="$RUN_DIR/vytallink.pid"
LOG_OUT="$ROOT/logs/app.out"
mkdir -p "$RUN_DIR" "$ROOT/logs"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi
if ! "$PY" -c "import vytallink" >/dev/null 2>&1; then
  echo "ERROR: vytallink not importable. Run scripts/setup.sh." >&2
  exit 1
fi

# Prevent duplicate launch.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    if ps -p "$OLD_PID" -o args= 2>/dev/null | grep -q "vytallink"; then
      echo "VytalLink already running (PID $OLD_PID). Use scripts/stop.sh first." >&2
      exit 1
    fi
  fi
  rm -f "$PID_FILE"  # stale
fi

# Resolve host/port for the address banner (defaults match settings).
PORT="$("$PY" -c "from vytallink.config import get_settings; print(get_settings().port)" 2>/dev/null || echo 5050)"

echo "==> Starting VytalLink..."
nohup "$PY" -m vytallink.app >>"$LOG_OUT" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"

# Wait for readiness.
READY=0
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "ERROR: process exited during startup. Last log lines:" >&2
    tail -n 20 "$LOG_OUT" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: server did not become healthy in time. See $LOG_OUT" >&2
  exit 1
fi

echo "==> VytalLink is running (PID $APP_PID)."
echo "    Local:   http://127.0.0.1:${PORT}"
LAN_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192|10|172)\.' | head -1 || true)"
if [[ -n "${LAN_IP}" ]]; then
  echo "    Network: http://${LAN_IP}:${PORT}  (same Wi-Fi/LAN)"
fi
echo "    Logs:    $LOG_OUT  and  logs/vytallink.log"
echo "    Stop:    scripts/stop.sh"
