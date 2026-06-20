#!/usr/bin/env bash
# Phase 15: repeated REAL start.sh -> stop.sh cycles in the production yolo/MPS,
# two-camera config — the exact path the Apple-MPS startup abort lived in.
# Records per cycle: launch attempts, whether a transient abort was retried,
# healthy+ready outcome, and full cleanup (PID gone, port free, no stale PID).
# Usage: diagnostics/run_startup_stress.sh [cycles]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CYCLES="${1:-10}"
PORT=5050

clean1=0; recovered=0; unrecovered=0; cleanup_ok=0; aborts_total=0
for i in $(seq 1 "$CYCLES"); do
  out="$(bash scripts/start.sh 2>&1)"; rc=$?
  attempts="$(printf '%s\n' "$out" | grep -c 'launching app')"
  aborts="$(printf '%s\n' "$out" | grep -c 'transient MPS startup abort')"
  aborts_total=$(( aborts_total + aborts ))
  if [[ $rc -eq 0 ]]; then
    state="$(curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('model',{}).get('state'))" 2>/dev/null)"
    if [[ "$attempts" -le 1 ]]; then clean1=$((clean1+1)); else recovered=$((recovered+1)); fi
    bash scripts/stop.sh >/dev/null 2>&1
    sleep 0.5
    pidgone=1; ps -p "$(cat run/vytallink.pid 2>/dev/null || echo 0)" -o pid= >/dev/null 2>&1 && pidgone=0
    portfree=1; [[ -n "$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null)" ]] && portfree=0
    nopidfile=1; [[ -f run/vytallink.pid ]] && nopidfile=0
    if [[ $portfree -eq 1 && $nopidfile -eq 1 ]]; then cleanup_ok=$((cleanup_ok+1)); fi
    printf "cycle %2d: OK attempts=%s aborts=%s model=%s cleanup(port_free=%s pidfile_gone=%s)\n" \
      "$i" "$attempts" "$aborts" "$state" "$portfree" "$nopidfile"
  else
    unrecovered=$((unrecovered+1))
    printf "cycle %2d: FAIL rc=%s attempts=%s aborts=%s\n" "$i" "$rc" "$attempts" "$aborts"
    rm -f run/vytallink.pid 2>/dev/null
  fi
done
echo "------------------------------------------------------------"
echo "cycles=$CYCLES clean_first_attempt=$clean1 recovered_by_retry=$recovered unrecovered_failures=$unrecovered"
echo "total_transient_aborts_observed=$aborts_total full_cleanup_ok=$cleanup_ok/$CYCLES"
