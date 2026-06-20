# Apple-silicon startup, model lifecycle, retry & related operations

This documents the hardening added on the `hardware-integration` branch after
multi-camera RTSP support: the Apple-MPS startup abort root cause + fix, bounded
startup retry, model-lifecycle health, the `/latest` alias, and synthetic
fall-test safety. It does not contain any RTSP URLs or credentials.

## 1. The intermittent MPS startup abort

**Symptom.** Occasionally (~1 in 5 launches) the process aborted right after
`Application startup complete` with:

```
-[_MTLCommandBuffer addScheduledHandler:]:807: failed assertion
  `Scheduled handler provided after commit call'   →   Abort trap: 6
```

**Root cause (proven).** Apple Metal command buffers are not safe across threads.
All model inference already runs on one dedicated inference thread. But the
`/health` endpoint called `gpu_info() → device_report() → select_device()`, which
**probed** the accelerator by creating a real MPS tensor and synchronizing — on
the event-loop thread. When the `/health` readiness probe landed while a camera
worker was inferring on the inference thread, two threads issued MPS command
buffers concurrently and Metal aborted the process. The Metal assertion is a C++
`abort()`, so it cannot be caught in Python — only a process-level retry recovers.

A focused two-thread reproduction (`scripts/mps_race_probe.py`: inference on
one thread, `gpu_info()` on another) aborted **4/10** before the fix and **0/15**
after.

**Fix (structural).** In `common/device.py`:
- The accelerator probe (the only op that creates a tensor / command buffer) is
  memoized behind a lock, so it runs **exactly once per process** — on the
  inference thread that resolves the device during startup.
- `device_report()` (and therefore `/health`/`gpu_info()`) no longer probes; it
  uses only static `is_available()`/`is_built()` checks plus the device the
  inference path published. No accelerator command buffer is ever created off the
  inference thread.

CUDA (Jetson) and CPU paths are unchanged: CUDA is still probed on the inference
thread, and `device_report()` reports CUDA via static checks.

**Stress harness.** `scripts/run_mps_race.sh [trials] [seconds]` runs the
in-process reproduction repeatedly and counts aborts. Use it to compare failure
rates before/after a change.

## 2. Bounded startup retry (`scripts/start.sh`)

`start.sh` now: preflights config + model file (fail fast, no retry), refuses to
start over a healthy instance or an unrelated process on the port, then launches
with bounded retry and a strict readiness gate.

- Success requires: process alive **and** port listening **and** `/health` 200
  **and** `model.state == ready`.
- Retries only a transient pre-health failure (e.g. the MPS abort, rc 134). It
  does **not** retry permanent errors (bad config, missing model) and never kills
  an unrelated process.
- Between attempts it confirms the prior PID is dead and the port is free, removes
  only a verified-stale PID file, and backs off (immediate, then 2s, then 5s).
- Ctrl-C during the retry sequence stops the child and exits cleanly.

Tunables (conservative defaults in `.env.example`):
`STARTUP_MAX_ATTEMPTS=3`, `STARTUP_RETRY_INITIAL_SECONDS=2`,
`STARTUP_RETRY_MAX_SECONDS=5`, `STARTUP_HEALTH_TIMEOUT_SECONDS=45`,
`MPS_STARTUP_STABILIZATION_SECONDS=0`.

Stop cleanly with `scripts/stop.sh` (graceful SIGTERM; removes the PID file).

## 3. Model lifecycle in `/health`

`/health` now includes:

```json
"model":   {"state": "ready", "device": "mps", "load_count": 1,
            "warmup_complete": true, "last_error": null},
"startup": {"attempt": 1, "max_attempts": 3, "completed": true, "uptime_seconds": 2.2}
```

`model.state` ∈ `loading | ready | degraded | failed`. The model is never `ready`
before warmup; a load failure surfaces as `failed` and (in live mode) degrades
overall health, so consumers never treat an unusable model as ready.

## 4. `/latest` compatibility alias

The canonical latest-vitals endpoint is **`/api/vitals/latest`**. For the legacy
iPhone vitals relay (which polled `GET /latest`), `/latest` is now a
backward-compatible alias returning the identical payload, schema, and no-vitals
behavior. Both routes call the same service function (no duplicated logic).

## 5. Synthetic fall testing (DEV-ONLY, safety-gated)

To validate the live persist→alert pipeline without staging a real fall you can
treat a present posture as fall evidence. This is gated to fail closed:

- Active when `SYNTHETIC_FALL_TEST_MODE=true` **or** a non-fall posture (e.g.
  `standing`, `sitting`) appears in `FALL_CLASS_NAMES`.
- **Rejected in production**; in development it requires
  `ALLOW_SYNTHETIC_FALL_TESTING=true` (otherwise startup fails closed).
- When active: external (webhook) alerts are forced to **dry-run**, every event
  is persisted as `event_type='fall_synthetic'`, a prominent startup warning is
  logged, and `/health` reports `synthetic_detection_mode: true`.

Clean up the marked events from the local dev DB afterward:

```bash
./.venv/bin/python scripts/cleanup_synthetic_events.py            # list (dry run)
./.venv/bin/python scripts/cleanup_synthetic_events.py --confirm  # delete
```

It only ever touches rows tagged `fall_synthetic` — real events
(`event_type='fall'`) can never be deleted by it, and it refuses to run against a
production environment.

## 6. Troubleshooting

- **Rare MPS abort still on first attempt:** the bounded retry recovers it; check
  `[startup]` lines in `logs/app.out`. Run `scripts/run_mps_race.sh` to
  measure the in-process rate.

## 7. Diagnostic harnesses (in `scripts/`, output is gitignored)

- `scripts/run_mps_race.sh [trials] [secs]` — in-process MPS cross-thread race
  reproduction (inference vs `gpu_info()`); counts aborts. Before fix ~4/10,
  after fix 0/15.
- `scripts/startup_stress.sh [cycles]` — repeated real `start.sh`→`stop.sh`
  cycles (yolo/MPS, two cameras); reports clean-first-attempt / retry-recovered /
  unrecovered and verifies PID + port cleanup each cycle.
- `scripts/soak_monitor.py [seconds] [interval]` — samples `/health` and process
  RSS to `diagnostics/soak.jsonl` and prints a begin/mid/end summary (memory,
  reconnects, queue depth, model-load count, tick errors).
- **`start.sh` exits 2/3/4 with no retry:** 2 = invalid config, 3 = missing model
  file, 4 = port held by another process. These are permanent; fix and re-run.
- **Stale PID / port held by a dead run:** `start.sh` removes a verified-stale PID
  file automatically; if a process is alive but unhealthy, run `scripts/stop.sh`.
- **Known limitation:** the abort is upstream Apple-MPS behavior; the fix removes
  the in-app cross-thread trigger and the retry covers any residual transient. It
  does not exist on the Jetson CUDA/TensorRT path.
