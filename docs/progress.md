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
- [x] **M4 — Providers (camera, detector, wearable, alerts)**
  - Camera: provider ABC (fps, stale detection, bounded-backoff reconnection),
    simulated camera, file + RTSP adapters (lazy cv2, credential-safe).
  - Detector: ABC + evidence mapping, simulated detector (scenarios/scripts),
    dormant YOLO adapter (lazy, clear errors, no downloads).
  - Wearable: ABC + deterministic simulated wearable (HR/motion/battery/conn).
  - Alerts: console + signed webhook (HMAC) providers + dispatcher (records
    every attempt, isolates provider exceptions).
  - Tests: camera (6) + detector (6) + wearable (6) + alerts (7) = 25.
  - Commit: `Add modular simulated monitoring providers`
- [x] **M5 — Backend API + health**
  - MonitoringService orchestrates providers → event pipeline: simulation mode
    drives the real pipeline deterministically (ManualClock) + health-only
    heartbeat; live mode uses a real-time detection loop. Wearable loop in both.
    Graceful start/stop, device registration, provider-exception isolation.
  - system_info (CPU/mem/disk-warning/GPU). FastAPI app: /health, /api/status,
    events (list/detail/label/resolve), devices, vitals (latest/list),
    simulation controls (dev+sim only), JSON error handling (no stack traces).
  - app.py entrypoint (uvicorn + graceful shutdown). Package installed editable.
  - Verified live via uvicorn+curl: 1 fall → 1 event + 1 alert, vitals, SIGTERM.
  - Tests: 15 API integration tests (85 total, all green).
  - Commit: `Add VytalLink monitoring API`
- [x] **M6 — Dashboard**
  - Responsive vanilla HTML/CSS/JS dashboard (no frameworks), mobile-friendly,
    polls /health + /api/status + /api/events + /api/devices every 3s. Shows
    overall health, fall state, camera/detector/wearable/GPU/DB status, vitals
    (HR/motion/battery/link), recent events with confidence/label/alert status,
    resolve + real-fall/false-alert actions, device warnings, simulation
    indicator, last-update time, nonmedical disclaimer. Dev controls
    (fall/normal/reset) shown only when controls_enabled. No live video.
  - Tests: 3 dashboard integration tests (88 total, all green).
  - Commit: `Add responsive caregiver dashboard`
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
