# VytalLink — Phase 1

VytalLink is a monitoring platform that combines **camera-based fall detection**,
**wearable vitals**, **event verification**, **caregiver alerts**, and a simple
**dashboard**. This repository is the Phase 1 implementation.

> ⚠️ **Important — not a medical device.** VytalLink is **not** a certified
> medical device, **not** an emergency service, and **not** a replacement for
> human supervision. Do not rely on it for safety-critical or life-critical
> monitoring. It is an experimental system under active development.

---

## 1. Purpose

Provide a reliable, privacy-respecting foundation for detecting possible falls,
confirming them as events (rather than reacting to every frame), notifying
caregivers once per event, and presenting status on a phone- or
computer-friendly dashboard — on a NVIDIA Jetson Orin Nano.

## 2. Phase 1 scope

Phase 1 is **simulation-first**. Everything works end-to-end using deterministic
simulated providers; the real-hardware paths are implemented as clean adapters
that stay dormant until real configuration/weights are supplied.

| Capability | Phase 1 status |
| --- | --- |
| Fall event state machine (confirm/recover/cooldown) | ✅ implemented + tested |
| SQLite persistence (events, vitals, alerts, devices) | ✅ |
| Simulated camera / detector / wearable | ✅ (real providers, labeled "simulation") |
| Console alerts | ✅ works with no credentials |
| Webhook alerts (HMAC-signed) | ✅ implemented (optional) |
| FastAPI backend + health | ✅ |
| Responsive dashboard | ✅ |
| RTSP camera adapter | 🟡 implemented, dormant (needs `CAMERA_SOURCE`) |
| YOLO model adapter | 🟡 implemented, dormant (needs `MODEL_PATH` + CUDA torch) |
| TensorRT export | ⛔ deliberately deferred until the real model is validated |
| Real wearable integration | ⛔ device not yet selected (see hardware docs) |

## 3. Architecture (summary)

```
camera (sim/file/rtsp) ─▶ detector (sim/yolo) ─▶ evidence ─▶ fall state machine
                                                              │
wearable (sim) ─▶ vitals ─▶ SQLite ◀── events/alerts/devices ─┤
                                                              ▼
                                          alert dispatcher (console + webhook)
                                                              │
                              FastAPI  ◀── MonitoringService ─┘
                                 │
                              Dashboard (polling) + JSON API
```

Layers: `common` → `config` → `database` → `events` → providers
(`vision`, `wearable`, `alerts`) → `monitoring` → `api` → `dashboard`.
Full detail in [`docs/architecture.md`](docs/architecture.md).

## 4. Environment

- NVIDIA **Jetson Orin Nano** Dev Kit, **JetPack 6.0** (L4T R36.3), Ubuntu 22.04,
  Python 3.10, CUDA 12.2, TensorRT 8.6, OpenCV 4.11.
- Project uses a **`--system-site-packages` venv** (keeps the Jetson-tuned
  `cv2`/`torch`/`tensorrt`) and installs only small pure-Python deps.
- System PyTorch is currently **CPU-only**; the CUDA wheel is a manual step
  before real inference. See [`docs/environment_decision.md`](docs/environment_decision.md)
  and [`docs/hardware_needed.md`](docs/hardware_needed.md).

## 5. Setup

```bash
cd ~/VytalLink
scripts/setup.sh          # creates .venv, installs deps, dirs, .env, inits DB
```

`setup.sh` is safe to run repeatedly.

## 6. Configuration

Configuration is environment-based. Copy and edit `.env` (created by `setup.sh`
from `.env.example`). Key variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VYTALLINK_ENV` | development / testing / production | development |
| `VYTALLINK_HOST` / `VYTALLINK_PORT` | bind address / port | 0.0.0.0 / 5050 |
| `VYTALLINK_DATABASE_PATH` | SQLite path (blank = default) | `data/database/vytallink.db` |
| `VISION_MODE` | simulation / file / rtsp | simulation |
| `CAMERA_SOURCE` / `CAMERA_USERNAME` / `CAMERA_PASSWORD` | camera config | (empty) |
| `DETECTOR_MODE` | simulation / yolo / tensorrt | simulation |
| `MODEL_PATH` | fall model weights | (empty) |
| `CONFIDENCE_THRESHOLD` | min fall confidence | 0.55 |
| `FALL_CONFIRM_SECONDS` | sustained evidence to confirm | 2.0 |
| `FALL_CLEAR_SECONDS` | evidence-absent to resolve | 3.0 |
| `ALERT_COOLDOWN_SECONDS` | min gap between alerts | 30 |
| `WEARABLE_MODE` | simulation | simulation |
| `WEBHOOK_URL` / `WEBHOOK_SECRET` | optional signed alerts | (empty) |
| `SAVE_EVENT_SNAPSHOTS` / `SAVE_EVENT_CLIPS` | event media (privacy: off) | false |

**Secrets live only in `.env`** (gitignored). They are never logged or returned
by the API; RTSP URLs are redacted before logging.

## 7. Simulation (Phase 1 default)

With `VISION_MODE=simulation` the system runs the full real pipeline
(camera → detector → state machine → alerts) but the fall timeline is driven
deterministically by the dev controls / API, so a fall confirms instantly.

- Dashboard buttons: **Simulate fall**, **Simulate normal**, **Reset**
  (visible only in development + simulation).
- API: `POST /api/simulation/fall`, `/normal`, `/reset`.

## 8. Real camera configuration (when available)

Set in `.env`:

```
VISION_MODE=rtsp
CAMERA_SOURCE=rtsp://CAMERA_HOST:554/Streaming/Channels/101
CAMERA_USERNAME=...
CAMERA_PASSWORD=...
```

The RTSP adapter uses TCP with a bounded open timeout and bounded-backoff
reconnection. See [`docs/hardware_needed.md`](docs/hardware_needed.md).

## 9. Real model configuration (when available)

```
DETECTOR_MODE=yolo
MODEL_PATH=/path/to/fall_model.pt
FALL_CLASS_NAMES=fall,fallen,lying
```

Requires the Jetson CUDA PyTorch wheel and `ultralytics` in the venv (the build
does **not** install these or download any model). TensorRT export comes only
after ordinary GPU inference is confirmed.

## 10. Start / stop

```bash
scripts/start.sh     # starts app, writes PID, prints local + LAN dashboard URLs
scripts/stop.sh      # graceful shutdown of only this project's process
```

Or run directly: `./.venv/bin/python -m vytallink.app`.

## 11. Dashboard access

- Local: `http://127.0.0.1:5050`
- LAN: `http://<jetson-ip>:5050` (printed by `start.sh`)

The dashboard polls the API every 3s. It does **not** show a live video feed.

## 12. Tests

```bash
./.venv/bin/python -m pytest        # full suite (deterministic, no real sleeps)
scripts/diagnose.sh                 # environment/import/DB/config/port/GPU/disk
scripts/smoke_test.sh               # full end-to-end workflow against a live server
```

## 13. Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named vytallink` | run `scripts/setup.sh` (installs the package editable) |
| Port already in use | another instance running — `scripts/stop.sh`, or change `VYTALLINK_PORT` |
| GPU shows unavailable | expected in Phase 1 (CPU-only torch); see hardware docs |
| Camera DOWN in live mode | verify `CAMERA_SOURCE`/credentials; check `logs/vytallink.log` (URLs redacted) |
| Config error at startup | the message names the missing/invalid variable |
| Webhook alert failing | recorded in DB with the failure reason; the app keeps running |

## 14. Privacy & security

- No live video stream is exposed; event snapshots/clips are **off by default**.
- Secrets only in `.env` (gitignored); never logged or returned by the API.
- RTSP credentials are redacted in all logs and health output.
- The database with real data, footage, model weights, and logs are gitignored.

## 15. Current limitations

- Hardware paths (RTSP, GPU model, real wearable) are implemented but untested
  against real devices in Phase 1 — they are marked **pending**.
- Single-node SQLite; no clustering, auth, or TLS (LAN/dev use).
- The dashboard has no authentication — keep it on a trusted LAN for now.

## 16. Roadmap

1. **Hardware enablement** — connect a real RTSP camera; install CUDA PyTorch +
   `ultralytics`; validate a real fall model on GPU.
2. **TensorRT** — export/validate an engine once GPU inference is confirmed.
3. **Wearable** — select a device, implement its provider behind the existing
   interface, store/scale vitals.
4. **Notifications** — add SMS/email/push providers behind the alert interface.
5. **Hardening** — dashboard auth, TLS, pilot deployment, observability.

See [`docs/pilot_checklist.md`](docs/pilot_checklist.md) before any real-world test.
