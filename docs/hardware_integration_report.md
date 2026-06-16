# VytalLink Hardware Integration Report

**Branch:** `hardware-integration` (do **not** merge to `main` until validated)
**Started:** 2026-06-16
**Goal:** identify the real legacy fall pipeline, enable safe Jetson GPU inference,
and integrate one real RTSP camera into the new provider architecture **without
breaking simulation mode or any existing test**.

This document is updated as each phase completes. Honest results only — a claim of
"works" appears only after the corresponding operation actually succeeded.

---

## Environment (verified)

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Orin Nano Developer Kit (`aarch64`) |
| L4T | R36.3.0 (`nvidia-l4t-core 36.3.0-20240719161631`) = JetPack 6.0 |
| CUDA toolkit | 12.2 (`nvcc V12.2.140`), `/usr/local/cuda-12.2` |
| TensorRT | 8.6.2 (present in venv) |
| Python | 3.10.12 (project venv `--system-site-packages`) |
| RAM | 7.4 GiB total (~5 GiB available at baseline) |
| Disk | ~205 GB free (12% used) |
| torch (baseline) | **2.6.0+cpu** in `~/.local/...` — `torch.cuda.is_available() == False` |
| torchvision | 0.21.0 · ultralytics 8.0.190 · opencv 4.11.0 · numpy 1.22.2 |

`torch`/`torchvision`/`ultralytics` live in `~/.local/lib/python3.10/site-packages`
and are visible because the venv was created with `--system-site-packages`. This
matters for the GPU plan (Phase 5): a GPU build installed **into the venv** shadows
the user-site CPU build, and rollback is simply uninstalling it — the system/user
torch is never touched.

---

## Phase 1 — Baseline & branch safety  ✅

| Check | Result |
|---|---|
| Branch | `hardware-integration` (created from clean `main`) |
| Working tree | clean at branch creation |
| `scripts/diagnose.sh` | **exit 0** (`Overall: WARN`) — WARNs: port 5050 in use (an app instance is running), GPU CUDA unavailable. Both expected; neither is a hard failure. |
| `./.venv/bin/python -m pytest` | **103 passed in 7.43 s** |
| `scripts/smoke_test.sh` | **PASS** — all 21 checks (start, health, dashboard, one-event/one-alert, duplicate suppression, label, resolve, persistence across restart, clean shutdown) |

Baseline is green. Proceeding.

---

## Phase 2 — Real legacy fall pipeline  ✅

Full evidence in **`docs/model_candidate_report.md`**. Summary:

* The canonical legacy pipeline is **`/project/rudra/vytalinkv1`** (same product as
  this repo). The separate `/project/rudra/fall-detection` directory is an older
  pose-based "visiage"/Azure prototype (it also contains a hard-coded Azure key that
  was deliberately **not** copied).
* **Model used:** `vytalinkv1/models/fall_detection.pt` — a **custom YOLO11n
  detection** model, classes `0=fallen, 1=sitting, 2=standing`; a fall is class 0
  `fallen`. (Confirmed: the file fails to load on `ultralytics 8.0.190` precisely
  because it uses the YOLO11-only `C3k2` module.)
* **Detector type:** custom fall-class detector — **not** pose + keypoint heuristics.
  The pose models (`yolov8n-pose.pt`) were used by v1 only for an *unused* activity
  side-channel.
* **Thresholds / timing (from `test_fall_fast.py`, the production live detector):**
  `imgsz=416`, `conf=0.55`, `infer_every=3`; temporal confirmation via an
  upright→fallen transition (`min_upright_frames=3`, `min_fallen_frames=3`,
  transition ≤ 2.5 s); confirm gate posture ≥ 0.75 / transition ≥ 0.50 / event ≥ 0.72;
  4 s `FallHold`; history cleared after 2 s with no detection; explicit out-of-frame
  handling; latest-frame threaded RTSP grabber with `BUFFERSIZE=1`.
* **Alert behavior:** per-frame `fall_score` fused with vitals into a 0–100 alert
  score (`fusion.py`); fall-safety overrides at 0.60 / 0.80.

These map cleanly onto the new architecture (the new defaults already match
`conf=0.55`, `imgsz=416`, `infer_every=3`); see the mapping table in the model report.
The new **state machine** plays the role of v1's sustained-fallen confirmation, and
v1's DTS-lite **transition gate** is reproduced as an optional detector module so we
do **not** pretend a pose model has a fall class and do **not** redesign the working
event/alert/dashboard layers.

---

## Phase 3 — Model candidate inspection  ✅

See `docs/model_candidate_report.md` for the full table, SHA256 hashes, load results,
and per-candidate classification. Selected: **`fall_detection.pt`** (evidence-based).

---

## Phase 4 — Selected model prepared  ✅

* `models/fall_detection.pt` copied from the legacy project (SHA256
  `3f56ad30…`, matches source). The `models/` dir is **git-ignored**
  (`models/*`, `*.pt`); `git check-ignore` confirms the model is not tracked.
* New command **`python -m vytallink.vision.test_model`** loads the model through
  the real adapter and reports sanitized name, task, classes, device, CUDA,
  warmup + inference latency. Verified output:
  `task=detect, classes={0:'fallen',1:'sitting',2:'standing'}, device=cuda:0`.
  This is the first time the model's classes were confirmed by **actually loading
  it** (ultralytics ≥ 8.3) — they match the legacy source exactly.

## Phase 5 — Jetson GPU inference  ✅ (CUDA + real inference verified)

**Approach chosen:** the **Ultralytics-hosted NVIDIA Jetson wheels** (vendor
Jetson path; `github.com/ultralytics/assets/releases/download/v0.0.0/`). The
`jetson-ai-lab` pip index was unreachable and the NVIDIA redist directory listing
now redirects to a wiki, so this was the official, reachable source. Picked the
**CUDA-12.2** pair matching JetPack 6.0 (the 2.5/2.10 builds target CUDA
12.4/12.6 = JP 6.1/6.2 and risk a "driver insufficient" error on R36.3):

* `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
* `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
* `ultralytics>=8.3,<8.4` (8.3.253) — required for the YOLO11 `C3k2` module.

All installed **into the venv with `--no-deps`**, which shadows the user-site
CPU build. The `~/.local` `torch 2.6.0+cpu` is **untouched** (pip explicitly
refused to remove it), so rollback is trivial.

**Verification (actually executed — not assumed):**
```
torch 2.3.0   cuda_available: True   cuda_build: 12.2
device: Orin  capability: (8,7)
torch.ones(4, device="cuda")*2 = [2,2,2,2]   (on cuda:0)
512x512 matmul + cuda.synchronize() : OK
torchvision.ops.nms on cuda          : [0,2]   (OK)
fall_detection.pt real inference     : ~33 ms/frame steady-state on cuda:0
```
fp16 is **disabled** by default: on this cuDNN 8.6 + torch 2.3 the YOLO11 conv
plans raise `CUDNN_STATUS_NOT_SUPPORTED` in half precision and fall back to a
~700 ms/frame path. fp32 runs ~30 fps, more than enough at `infer_every=3`.

### Rollback plan (dependency changes)

Baseline captured to `/tmp/pip_freeze_baseline.txt` before any change. Because
the GPU build lives only in the venv:

```bash
# Full rollback to the CPU baseline (re-exposes ~/.local torch 2.6.0+cpu):
./.venv/bin/pip uninstall -y torch torchvision ultralytics
# (optional) recreate the venv from scratch:
rm -rf .venv && scripts/setup.sh
```
No JetPack/CUDA/cuDNN/driver/system-Python change was made. Re-download URLs for
the exact wheels are recorded above.

## Phase 6 — Secure RTSP configuration  ✅

* New component fields: `CAMERA_HOST`, `CAMERA_PORT` (554), `CAMERA_STREAM_PATH`,
  plus separate `CAMERA_USERNAME` / `CAMERA_PASSWORD`. A full URL in
  `CAMERA_SOURCE` still wins. The URL is assembled **in memory**
  (`Settings.rtsp_url()`), credentials URL-encoded (so an email username / a
  password with `@:/` is safe), and **only the redacted form is ever logged**.
* `.env` (git-ignored) holds the placeholder camera + creds; `.env.example`
  documents the fields. App default stays **simulation** so nothing breaks —
  flip `VISION_MODE=rtsp` + `DETECTOR_MODE=yolo` to go live.
* Verified: assembled URL contains host/port/path; `sanitized_camera_source()`
  and `safe_summary()` redact username + password (asserted in code + tests).

## Phase 7 — Camera diagnostics  ✅

`python -m vytallink.vision.test_camera` — connects, runs ≥60 s (configurable),
**saves no footage, opens no window**, reports resolution / frames / effective
FPS / failed reads / reconnects / stale warnings / avg read latency, uses bounded
reconnection, shuts down cleanly, returns non-zero on failure, sanitizes errors.
Validated against the placeholder host (no camera present): source shown as
`rtsp://***REDACTED***@192.168.42.251:554/Streaming/Channels/101`, graceful
failure, `RESULT: CAMERA_FAIL`, exit 1 — i.e. it fails safely and leaks nothing.

## Phase 8 — Live inference integration  ✅

* **`RTSPCamera`** now runs a daemon **latest-frame grabber** thread
  (`BUFFERSIZE=1`, FFmpeg TCP, bounded open timeout) — readers always get the
  freshest frame; stale frames are intentionally dropped (counted). This is the
  legacy v1 `FrameGrabber` design, reproduced cleanly.
* **`YoloFallDetector`** loads the model **once**, resolves the real device,
  warms up, and records inference latency / FPS / device / count. Camera read +
  inference already run **off the FastAPI event loop** (`asyncio.to_thread` in
  `MonitoringService`), so a slow/bad RTSP source can never freeze the API.
* Configurable: confidence, image size, `PROCESS_EVERY_N_FRAMES`, fall classes,
  confirm/clear/cooldown, plus new `DETECTOR_REQUIRE_TRANSITION` and
  `EVIDENCE_HOLD_SECONDS`.
* **`python -m vytallink.vision.live_detection`** — standalone diagnostic that
  prints sanitized detections + metrics and **suppresses caregiver alerts unless
  `--alerts`**.

### Two real-data findings (and the fixes)

Running the real model over the legacy labelled clips (read-only) surfaced two
issues that the synthetic simulation could never show:

1. **Posture alone cannot tell a fall from "already lying down."** Both show an
   upright→fallen posture sequence; only *velocity* separates them (v1's full
   DTS). The strict transition gate also **missed real falls** because detection
   is sparse. → The `PostureTransitionGate` is now gap-tolerant and **OFF by
   default** (`DETECTOR_REQUIRE_TRANSITION=false`): a sustained `fallen` posture
   is the fall signal, so **no real fall is missed**. The gate remains an opt-in,
   tested filter (it rejects "already lying, never seen upright").
2. **Sparse detection breaks the state machine's continuous-evidence rule.** Real
   YOLO output drops frames mid-fall, which would reset a candidate event. →
   New live-only **`FallEvidenceSmoother`** bridges brief gaps (default 1.0 s,
   below `FALL_CLEAR_SECONDS`), so a sustained fall reads as continuous evidence;
   a clear upright cancels it immediately. The **state machine is unchanged**.

## Phase 9 — Event correctness  ✅

27 new tests (no hardware required) drive the **real detector path** (fake
weights) → `detections_to_evidence` → state machine / `EventManager`:
brief evidence → no event; sustained → exactly one event + one alert; continued
detections → no duplicate event/alert; recovery; later independent fall → second
event; failed alert delivery → next fall still alerts; sparse detection + smoother
→ confirms; transition gate rejects already-lying but confirms a real transition;
and the **simulation path still produces evidence**. Full suite: **136 passed**.

## Phase 10 — Dashboard & health  ✅

`/health` now carries `mode`, sanitized `camera_name`, and (via the camera/detector
dicts) camera status/FPS/resolution/reconnects/dropped, detector device/inference
FPS/latency/count, GPU device + compute capability, and last frame/inference time.
The dashboard gained a **Hardware** card and a **LIVE/SIMULATION** pill. It exposes
**no** camera username/password, no complete RTSP URL, and **no model filesystem
path** (basename only). Verified live (see Phase 11.9).

## Phase 11 — Validation (honest results)

| # | Check | Result |
|---|---|---|
| 1 | Existing unit/integration tests | **136 passed** (103 baseline + 33 new) |
| 2 | Simulation smoke test | **PASS** (all 21 checks; sim path unchanged) |
| 3 | CUDA verification | **OK** — real CUDA tensor op + matmul on Orin (`cuda:0`) |
| 4 | Model inspection | **OK** — `test_model` loads, classes `fallen/sitting/standing` |
| 5 | Synthetic model inference | **OK** — ~33 ms/frame fp32 on GPU |
| 6 | Camera connection test | Graceful **FAIL vs placeholder** (no camera); redacted, exit 1 |
| 7 | 60-second stability test | Tool verified (ran shorter vs placeholder); needs a real camera |
| 8 | Real-frame inference | **OK** — model detects `fallen` on real fall clips; `sitting`/empty → none |
| 9 | Real-mode health endpoint | **OK** — `mode=rtsp`, `device=cuda:0`, `gpu=Orin`, **no credential/path leak** |
| 10 | Dashboard responsiveness | **OK** — served + polling `/health`,`/api/status`; hardware card renders |
| 11 | Staged posture test | **Pending real camera/person** (do NOT fall forcefully; use a padded slow lie-down) |

**Real-frame results** (legacy clips, default pipeline, gate off, smoother on):

| clip type | model calls `fallen` | sustained run | confirms @1.0s | @2.0s |
|---|---|---|---|---|
| FALL ×3 | yes (18–32 frames) | 1.4–2.4 s | **CONFIRMED** | possible/confirmed |
| ALREADY_LYING | yes | 2.0 s | confirmed* | possible |
| SITTING | none | 0 | normal | normal |
| NO_PERSON | none | 0 | normal | normal |

\* already-lying confirming is the documented posture-only false positive; it is
mitigated by `FALL_CONFIRM_SECONDS`, the opt-in transition gate, and caregiver
labeling. The clips are short (the fall *moment*); a live stream where a fallen
person stays down sustains well past `FALL_CONFIRM_SECONDS`.

### Known limitations / remaining work

* **No live camera available** — the configured IPs/credentials are placeholders;
  the camera connection, 60 s stability, and staged-posture tests need real
  hardware. Everything up to the camera boundary is validated.
* **Already-lying vs fall** needs the velocity/trajectory DTS (v1's full
  approach) for robust discrimination; Phase-1 ships the safe "sustained fallen"
  default plus the opt-in posture gate.
* **fp16** disabled pending a cuDNN that supports the YOLO11 conv plans.
* **TensorRT** export intentionally not done yet (validate GPU first — done).
* **Wearable** remains simulation (no device selected).

## Phase 12 — Post-integration adversarial review  ✅

An 11-agent adversarial review swept the new code for: duplicate events,
duplicate alerts, failed alert delivery, stale frames, reconnect behavior,
event-loop blocking, credential leakage, model reloads, unbounded queues, false
health status, and simulation regressions.

**8 areas came back clean:** duplicate-events, duplicate-alerts,
failed-alert-delivery, event-loop-blocking, credential-leakage, model-reloads,
unbounded-queues, simulation-regressions. **6 real defects were found in the new
code and fixed** (each with a regression test):

| Severity | Defect | Fix |
|---|---|---|
| HIGH | RTSP reconnect race: a grabber thread blocked in `cap.read()` (5 s FFmpeg timeout) could outlive the 2 s join, then write a **stale frame as "fresh"** after the new connection reset state — and touch a **released capture** (use-after-free). | Generation-fenced grabber: each thread is bound to a per-connection generation and **owns/releases its own capture**; superseded writes are dropped under the lock; the capture is never released out from under an in-flight read. (`vision/rtsp.py`) |
| HIGH | Overall `/health` ignored the detector — a model failing **every** inference still reported `overall=ok`. | `health()` now folds detector status in: live + detector DOWN → DOWN; detector DEGRADED → DEGRADED; live + camera DEGRADED → DEGRADED. (`monitoring/service.py`) |
| HIGH | `YoloFallDetector` reported `status=ok` whenever the model object existed, even if every `infer()` raised. | Runtime status: DEGRADED when the last inference failed; `last_error` cleared on success. (`vision/detector_yolo.py`) |
| MED | `latest_inference_time` advanced even on failed inference (false "pipeline alive"). | Only advanced on a successful inference. (`monitoring/service.py`) |
| LOW | `frames_dropped` could go negative (clamped) when a reader out-paced the grabber. | `frames_consumed` counts only genuinely new frames. (`vision/rtsp.py`) |
| LOW | `safe_summary()` logged the absolute model path locally (not a credential, but inconsistent with the "no absolute path" guarantee). | Logs the basename only. (`config/settings.py`) |

Two findings were correctly self-assessed as **non-bugs** and not "fixed":
a deliberate gpu_info process-cache, and the (now-addressed) model-path logging.

Final state after fixes: **141 tests pass**, smoke test PASS, real-mode health
endpoint verified leak-free, model inference on GPU confirmed.

---

## How to go live (operator quick-start)

1. `nano ~/VytalLink/.env` → set `VISION_MODE=rtsp`, `DETECTOR_MODE=yolo`, and the
   real `CAMERA_HOST`/`CAMERA_PORT`/`CAMERA_STREAM_PATH`/`CAMERA_USERNAME`/`CAMERA_PASSWORD`.
2. `./.venv/bin/python -m vytallink.vision.test_model`   — confirm GPU + model.
3. `./.venv/bin/python -m vytallink.vision.test_camera`  — confirm the camera is stable (≥60 s).
4. `./.venv/bin/python -m vytallink.vision.live_detection` — watch sanitized detections (no alerts).
5. `scripts/start.sh` — run the full app; the dashboard shows the LIVE hardware panel.


