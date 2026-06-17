# VytalLink — Apple-silicon (macOS) Development & Inference

This documents running VytalLink on an Apple-silicon Mac for development and
real model validation, alongside the primary Jetson Orin Nano deployment. The
Jetson/CUDA path is **unchanged**; macOS adds an Apple **MPS** inference path
and CPU fallback. Phase-1 simulation remains the default everywhere.

## Why a Mac path

The Jetson remains the deployment target. A Mac is useful for development and
for validating the real YOLO fall model on a GPU-class backend (Apple MPS)
without the Jetson CUDA wheel. Nothing here requires CUDA, TensorRT, JetPack,
`nvidia-smi`, `tegrastats`, or systemd; Linux-only checks skip cleanly on macOS.

## Verified environment

| | |
|---|---|
| OS / arch | Darwin (macOS) `arm64` |
| Python | 3.12.12 (project venv) |
| PyTorch | 2.12.0 |
| CUDA available | False |
| MPS built / available | True / True |
| Ultralytics | 8.4.69 |
| Selected inference device | **mps** |

## Setup commands

```bash
# From the repo root. Creates .venv, installs deps, prepares dirs,
# creates .env from .env.example (never overwrites an existing .env),
# and initializes the dev database. Idempotent.
scripts/setup.sh

# Sanity check the environment (OS, arch, Python, Torch, CUDA, MPS,
# selected device, DB, ports — all sanitized).
scripts/diagnose.sh

# Run / stop the app (simulation by default).
scripts/start.sh
scripts/stop.sh
```

Notes:
- On Apple silicon, PyTorch + Ultralytics install as ordinary wheels (no
  `--system-site-packages` heavy libs are required as on the Jetson). The venv
  is created the same way for parity; the Mac simply has no system CUDA libs to
  inherit.
- `scripts/start.sh` prints a LAN address using `ipconfig getifaddr enN` on
  macOS (the Jetson uses `hostname -I`); both are best-effort and skip cleanly.

## Device-selection behavior

Device selection is centralized in `src/vytallink/common/device.py`
(`select_device`) so the detector, diagnostics, and health all agree. It
**probes** each backend with a real tensor op and falls back safely — a backend
that reports available but cannot execute is skipped, never crashing startup:

1. **CUDA** — `torch.cuda.is_available()` *and* a real CUDA tensor op succeeds
   (the Jetson path once the CUDA wheel is installed) → `cuda:0`.
2. **MPS** — CUDA unavailable, Apple `mps` available, *and* a real MPS tensor op
   succeeds (the Apple-silicon path) → `mps`.
3. **CPU** — otherwise → `cpu`.

An explicit preference (`DETECTOR` device arg) is honored only if it probes OK,
otherwise selection falls through to auto-select.

The selected device is visible in:
- `scripts/diagnose.sh` → the `inference_device` check,
- `GET /health` → `gpu.selected_device`, `gpu.mps_available`, `gpu.cuda_available`,
- `python -m vytallink.vision.test_model` (validation/benchmark).

Only device strings and boolean flags are exposed — never filesystem paths,
usernames, or host-private info. The DB and model are reported by **filename
only**.

### MPS → CPU fallback

If a YOLO/PyTorch op is unsupported on MPS, `YoloFallDetector` records the exact
error in `mps_fallback_reason`, moves the model to CPU, and continues on CPU. It
never silently claims MPS: `device` flips to `cpu` and the reason is surfaced in
health and the benchmark. On this machine the fall model runs fully on MPS with
**no fallback** (see below).

`fp16`/`half` is CUDA-only and off by default (Jetson cuDNN reasons); MPS uses
fp32.

## Model benchmark results (this Mac)

Run separately from the app (the app stays in simulation):

```bash
python -m vytallink.vision.test_model
```

Validated `models/fall_detection.pt`:
- **task**: `detect`
- **classes**: `{0: fallen, 1: sitting, 2: standing}` — expected `fallen`,
  `sitting`, `standing` all present
- **device**: `mps` (half=False), **no CPU fallback**
- synthetic-image inference succeeds; the model is loaded once and reused.

Benchmark (synthetic 416×416 frames, 1 warmup + 15 measured steady-state,
MPS synchronized before each timing):

| metric | value |
|---|---|
| image size | 416×416 |
| warmup | ~12.7 ms |
| average | ~10.5 ms |
| median | ~10.8 ms |
| min / max | ~8.1 ms / ~11.3 ms |
| approx FPS | ~95 |

(Exact numbers vary run to run; re-run the command to reproduce.)

Ultralytics is configured with `save=False`/`save_txt=False`, so inference never
writes annotated frames or labels into `runs/` (footage stays off disk).

## Known Mac limitations

- **CUDA / TensorRT are unavailable** on macOS — the `tensorrt` detector mode is
  Jetson-only and stays not-implemented in Phase 1.
- **MPS is fp32 only here**; no fp16 path.
- **Some torch/YOLO ops may be unsupported on MPS** depending on the model. The
  detector falls back to CPU and reports it rather than failing.
- The Mac dev disk may run low; `GET /health` correctly reports `disk_warning`
  and degrades overall health at the production 90% threshold. We do **not**
  lower the production threshold for low disk — tests/smoke raise
  `DISK_WARNING_PERCENT` only to stay independent of the host disk.
- Linux-only diagnostics (`nvidia-smi`, `tegrastats`, `hostname -I`) are not
  used on macOS; the scripts use portable alternatives or skip cleanly.

## Returning to Jetson deployment

No code changes are needed — device selection auto-detects CUDA. To run the
real detector on the Jetson:

1. Keep simulation defaults for normal operation, or set real-hardware modes in
   `.env` when validating hardware:
   - `VISION_MODE=rtsp` (+ camera config) — **only when intentionally enabling a
     real camera**; leave `simulation` otherwise.
   - `DETECTOR_MODE=yolo`, `MODEL_PATH=models/fall_detection.pt`.
2. Install the **Jetson CUDA PyTorch wheel** (see `docs/hardware_needed.md`); the
   system OpenCV/TensorRT remain inherited via `--system-site-packages`.
3. `scripts/diagnose.sh` should then report `inference_device … device=cuda:0`.
4. `python -m vytallink.vision.test_model` validates the model on CUDA.

The CUDA path is selected automatically ahead of MPS/CPU, so the same code runs
unchanged on the Jetson.
