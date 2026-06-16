# VytalLink Phase 1 — Build Progress

Live checklist updated as milestones complete. See `docs/morning_report.md`
for the final summary.

## Milestones

- [x] **M1 — Environment, repo, config, structure, docs**
  - System inspection (`docs/system_report.md`), environment decision
    (`docs/environment_decision.md`), git on `main`, `.gitignore`,
    pydantic-settings config with validation + secret sanitization, common
    building blocks (Clock, logging, types, errors), project structure.
  - Commit: `Initialize VytalLink Phase 1 foundation`
- [x] **M2 — Database and core models**
  - SQLite schema (events/vitals/alerts/devices) + indexes + migrations,
    thread-safe connection manager, parameterized repositories, row models.
  - Tests: config (10) + database (12). Commit: `Add persistent event and vital storage`
- [x] **M3 — Fall event state machine**
  - States normal/possible/confirmed/recovering/resolved; confirm/clear timing;
    cooldown; exactly-one-alert; duplicate suppression; manual resolve/reset;
    EventManager (persistence + alert dispatch). Clock-injected, sleep-free tests.
  - Tests: state machine (16) + event manager (8). Commit: `Implement tested fall event state machine`
- [ ] **M4 — Providers (camera, detector, wearable, alerts)**
- [ ] **M5 — Backend API + health**
- [ ] **M6 — Dashboard**
- [ ] **M7 — End-to-end validation + scripts**
- [ ] **M8 — Hardware adapters + final docs + morning report**

## Test status
- Full suite green: **45 passed** (as of M3).

## Key decisions
- `--system-site-packages` venv (keeps Jetson cv2/torch/tensorrt); FastAPI.
- Clock injection everywhere timing matters → deterministic tests + instant,
  real (non-mock) simulation driving.
- DB rows are created at **confirmation** (possible blips are not persisted),
  keeping the events table meaningful. Live current-state is in-memory.
- Simulation providers are *real* providers labeled "simulation" — never mocks
  in the production path.
