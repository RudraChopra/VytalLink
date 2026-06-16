# CLAUDE.md — VytalLink project rules & conventions

VytalLink is a monitoring platform: camera-based fall detection, wearable
vitals, event verification, caregiver alerts, and a dashboard. **Phase 1 is
simulation-first** — real camera/model/wearable paths exist as clean adapters
but are dormant until hardware config/weights are supplied.

## Safety & legal (non-negotiable)
- VytalLink is **NOT** a certified medical device, emergency service, or a
  replacement for human supervision. Every user-facing surface says so.
- Never commit secrets, footage, model weights, TRT engines, real databases,
  logs, caches, or venvs (see `.gitignore`). Secrets live only in `.env`.
- Sanitize credentials before logging (`common/sanitize.py`). RTSP URLs are
  redacted; no password/secret is ever logged or returned by the API.
- Do not expose a live camera feed via the dashboard. Event media is off by
  default (`SAVE_EVENT_SNAPSHOTS` / `SAVE_EVENT_CLIPS`).

## Environment
- Jetson Orin Nano, JetPack 6.0 (L4T R36.3), Python 3.10, CUDA 12.2, TRT 8.6.
- Project venv: `python3 -m venv --system-site-packages .venv` (keeps system
  `cv2`/`torch`/`tensorrt` available). See `docs/environment_decision.md`.
- System PyTorch is **CPU-only** today; CUDA wheel is a manual step before real
  inference (`docs/hardware_needed.md`). Do NOT modify JetPack/CUDA/drivers.

## Architecture conventions
- Source under `src/vytallink/`, installed/importable as `vytallink`.
- Layers: `common` (clock/types/logging/sanitize) → `config` → `database`
  (schema + repositories) → `events` (state machine + manager) → providers
  (`vision`, `wearable`, `alerts`) → `monitoring` (orchestration) →
  `api` (FastAPI) → `dashboard`.
- Type hints everywhere. UTC internally (`common/clock.py`). JSON API responses.
- **Clock injection**: timing logic takes a `Clock`. Tests use `ManualClock`
  (no real sleeps). Simulation driver also uses `ManualClock` to make the
  fall pipeline deterministic and instant.
- Providers are interfaces with simulated + hardware implementations. The
  simulated providers are *real* working providers, explicitly labeled
  `simulated`/`simulation` — never mocks in the production path.
- SQLite only (Phase 1). Parameterized SQL. One connection guarded by a lock.

## Commands
- Setup:        `scripts/setup.sh`
- Diagnose:     `scripts/diagnose.sh`
- Start:        `scripts/start.sh`   (writes PID, prints dashboard URLs)
- Stop:         `scripts/stop.sh`
- Smoke test:   `scripts/smoke_test.sh`
- Reset demo:   `scripts/reset_demo_data.sh`  (dev DB only)
- Tests:        `./.venv/bin/python -m pytest`        (from repo root)
- Run app:      `./.venv/bin/python -m vytallink.app` (or the `vytallink` entry)

## Testing rules
- Deterministic. No long real sleeps — inject/advance `ManualClock`.
- State-machine, config, DB, providers, API all have unit/integration tests.
- Run the full suite after each milestone. Don't reduce coverage to pass.

## Don'ts
- No Redis/Postgres/K8s/cloud. No new heavy deps without a clear reason.
- Don't put the whole app in one file. Don't leave fake "always-success" stubs
  in real paths. Don't download a random model and call it the VytalLink model.
