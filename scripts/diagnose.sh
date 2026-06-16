#!/usr/bin/env bash
# VytalLink diagnostics — environment, imports, DB, config, port, GPU, disk.
# Exits non-zero only if a hard check FAILs (WARN is allowed).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

exec "$PY" -m vytallink.diagnostics
