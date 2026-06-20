#!/usr/bin/env bash
# Run the in-process MPS race probe K times; count aborts (signal 6 / exit 134).
# Usage: scripts/run_mps_race.sh [trials] [seconds-per-trial]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
TRIALS="${1:-10}"
SECS="${2:-3}"
aborts=0; survived=0
for i in $(seq 1 "$TRIALS"); do
  "$PY" "$ROOT/scripts/mps_race_probe.py" "$SECS" >/tmp/mps_trial.out 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    survived=$((survived+1)); printf "trial %2d: OK (%s)\n" "$i" "$(grep -o 'SURVIVED.*' /tmp/mps_trial.out | head -1)"
  else
    aborts=$((aborts+1)); printf "trial %2d: ABORT rc=%d (%s)\n" "$i" "$rc" "$(grep -oiE 'abort trap|addScheduledHandler|MTLCommandBuffer' /tmp/mps_trial.out | head -1)"
  fi
done
echo "----------------------------------------"
echo "trials=$TRIALS survived=$survived aborts=$aborts"
