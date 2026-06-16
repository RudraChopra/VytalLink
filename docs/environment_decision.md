# VytalLink — Environment Decision

## Decision

**Use a project virtual environment created with `python3 -m venv --system-site-packages .venv`**, into which only a small set of pure-Python project dependencies are installed.

We did **not** choose Docker for Phase 1.

## Reasoning

The host is a **Jetson Orin Nano Developer Kit** running **JetPack 6.0 (L4T R36.3.0)**, **Python 3.10.12**, **CUDA 12.2**, **TensorRT 8.6.2**, with system **OpenCV 4.11.0** importable. See `docs/system_report.md` for the full inventory.

### Why a venv (not Docker)

1. **Phase 1 is simulation-only.** It performs no GPU inference, so it needs no CUDA container. The dependencies are a web framework, a settings library, an HTTP client, and a test runner — all pure Python.
2. **Reliability & iteration speed.** A venv starts instantly, is trivial to recreate, and does not require image pulls, GPU passthrough configuration, or root. Docker adds moving parts that Phase 1 does not benefit from.
3. **`--system-site-packages` preserves the Jetson-tuned heavy libraries.** On a Jetson you cannot simply `pip install` a CUDA-enabled PyTorch or the platform OpenCV — they ship as system packages built for L4T. By inheriting system site-packages, the venv can already `import cv2` (4.11.0), `import tensorrt` (8.6.2), and `import torch` — verified during setup. This means the future vision pipeline needs **no rebuild** of those libraries.
4. **Isolation of project deps.** FastAPI, uvicorn, pydantic, httpx, pytest, etc. install into the venv and take precedence over anything system-wide, without touching the system Python. None of them overlap with `cv2`/`torch`/`tensorrt`, so there is no version conflict.

### Why not install desktop PyTorch wheels

Per the operating boundaries, we do **not** install random PyTorch wheels. The system PyTorch is currently the **CPU-only** build (`2.6.0+cpu`, `cuda.is_available()==False`). Replacing it with the Jetson CUDA wheel is a deliberate, documented hardware-enablement step (see `docs/hardware_needed.md`), to be run manually before real model inference. Phase 1 does not require it.

### Web framework: FastAPI (not Flask)

We chose **FastAPI + uvicorn** because:

- **Built-in request/response validation** via pydantic models (the API spec requires body validation, correct status codes, and useful errors — FastAPI gives this declaratively).
- **Async** — the monitoring/wearable loops run as asyncio background tasks under uvicorn, keeping the API responsive while loops run (an explicit requirement).
- **JSON-native** responses and automatic OpenAPI docs.
- Strong type-hint integration, matching our typed codebase.

Flask would have required hand-rolled validation and a separate threading model for background work. Neither framework was pre-installed, so the "already installed" tiebreaker did not apply; reliability and fit decided it. `jinja2` (for the dashboard template) and `psutil` (system metrics) were already present system-wide and are reused via `--system-site-packages`.

## How to recreate

```bash
cd ~/VytalLink
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-dev.txt   # runtime + test
# or, runtime only:
./.venv/bin/python -m pip install -r requirements.txt
```

`scripts/setup.sh` automates exactly this and is safe to run repeatedly.

## Installed project dependencies (pinned)

Runtime: `fastapi==0.115.6`, `uvicorn==0.34.0`, `pydantic==2.10.4`, `pydantic-settings==2.7.0`, `python-dotenv==1.0.1`, `httpx==0.28.1`, `jinja2==3.1.6`, `psutil==7.0.0`.
Dev/test: `pytest==8.3.4`, `pytest-asyncio==0.25.0`.

Inherited from system (via `--system-site-packages`): `cv2 4.11.0`, `torch 2.6.0+cpu`, `tensorrt 8.6.2`, `numpy 1.22.2`.
