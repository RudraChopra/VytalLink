#!/usr/bin/env bash
# VytalLink stop — stops only the process this project started (by PID file),
# gracefully first (SIGTERM), then SIGKILL if needed. No broad process killing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PID_FILE="$ROOT/run/vytallink.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file; VytalLink does not appear to be running (started by this project)."
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "${PID}" ]] || ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID not running; removing stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi

# Safety: only kill if the process really is our app.
if ! ps -p "$PID" -o args= 2>/dev/null | grep -q "vytallink"; then
  echo "PID $PID does not look like VytalLink; refusing to kill. Remove $PID_FILE manually if stale." >&2
  exit 1
fi

echo "==> Stopping VytalLink (PID $PID) gracefully..."
kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then break; fi
  sleep 0.5
done

if kill -0 "$PID" 2>/dev/null; then
  echo "==> Did not stop in time; sending SIGKILL."
  kill -KILL "$PID" 2>/dev/null || true
  sleep 0.5
fi

rm -f "$PID_FILE"
echo "==> Stopped."
