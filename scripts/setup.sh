#!/usr/bin/env bash
# VytalLink setup — idempotent. Creates the venv, installs deps, prepares dirs,
# creates .env from the example, and initializes the dev database.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "pyproject.toml" || ! -d "src/vytallink" ]]; then
  echo "ERROR: must run from the VytalLink project root (pyproject.toml not found)." >&2
  exit 1
fi

VENV="$ROOT/.venv"
PY="$VENV/bin/python"

echo "==> VytalLink setup (root: $ROOT)"

# 1. Create venv with system site packages (keeps Jetson cv2/torch/tensorrt).
if [[ ! -x "$PY" ]]; then
  echo "==> Creating virtual environment (.venv, --system-site-packages)"
  python3 -m venv --system-site-packages "$VENV"
else
  echo "==> Reusing existing virtual environment"
fi

# 2. Install project dependencies (only the project's, into the venv).
echo "==> Upgrading pip and installing dependencies"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements-dev.txt
"$PY" -m pip install -e .

# 3. Create local runtime directories.
echo "==> Creating runtime directories"
mkdir -p data/database data/events data/clips logs run

# 4. Create .env from example only if it does not exist.
if [[ ! -f ".env" ]]; then
  echo "==> Creating .env from .env.example (edit it to add real config later)"
  cp .env.example .env
else
  echo "==> .env already exists; leaving it untouched"
fi

# 5. Initialize the development database (safe to repeat — migrations are idempotent).
echo "==> Initializing database"
"$PY" - <<'PYEOF'
from vytallink.config import get_settings
from vytallink.database import Database
s = get_settings()
db = Database(s.database_path)
v = db.initialize()
db.close()
print(f"   database ready at {s.database_path} (schema v{v})")
PYEOF

echo "==> Setup complete."
echo "    Next: scripts/diagnose.sh   then   scripts/start.sh"
