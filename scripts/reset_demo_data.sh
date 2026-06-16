#!/usr/bin/env bash
# VytalLink demo-data reset — clears events/vitals/alerts/devices from the
# DEVELOPMENT database inside this project only. Refuses to run in production
# and refuses to touch a database outside the project tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

"$PY" - <<PYEOF
import sys
from pathlib import Path
from vytallink.config import get_settings, Environment
from vytallink.database import Database, Repositories

s = get_settings()
root = Path("$ROOT").resolve()
db_path = Path(s.database_path).resolve()

if s.env == Environment.PRODUCTION:
    print("Refusing to reset data in PRODUCTION environment.")
    sys.exit(1)

if root not in db_path.parents and db_path != root:
    print(f"Refusing: database {db_path} is outside the project tree {root}.")
    sys.exit(1)

db = Database(db_path)
db.initialize()
for table in ("alerts", "vitals", "events", "devices"):
    db.execute(f"DELETE FROM {table}")
db.execute("VACUUM")
r = Repositories(db)
print(f"Reset demo data in {db_path}")
print(f"   events={r.events.count()} vitals={r.vitals.count()} alerts={r.alerts.count()}")
db.close()
PYEOF
