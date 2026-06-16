# VytalLink — Hardware & Configuration Needed Later

Everything below is what **you** must provide to move beyond Phase 1 simulation.
Nothing here is guessed or invented; the build never probes credentials or
downloads models. Put all secrets in `.env` only (never commit them).

## 1. RTSP camera

| # | Item | Example / notes |
| --- | --- | --- |
| 1 | RTSP address (`CAMERA_SOURCE`) | `rtsp://CAMERA_HOST:554/Streaming/Channels/101` (Hikvision-style) or your camera's documented path |
| 2 | Camera username (`CAMERA_USERNAME`) | a least-privilege viewing account |
| 3 | Camera password (`CAMERA_PASSWORD`) | stored only in `.env` |
| 4 | Stream path | the exact URL path/channel for the substream you want |

Then set `VISION_MODE=rtsp` and restart. The adapter uses TCP transport with a
bounded open timeout and reconnects with backoff. Verify reachability first,
e.g. `ffprobe "rtsp://.../..."` (do not paste credentials into shell history).

## 2. Fall model

| # | Item | Notes |
| --- | --- | --- |
| 5 | Fall model file (`MODEL_PATH`) | a trained model file on this Jetson |
| 6 | Model framework | Phase 1 adapter targets **Ultralytics YOLO** (`.pt`). TensorRT is added only after GPU inference is validated. |
| 7 | Expected model classes | the class name(s) that mean "fall" — set `FALL_CLASS_NAMES` to match (e.g. `fall,fallen,lying`) |

Then set `DETECTOR_MODE=yolo` and restart.

## 3. GPU / PyTorch enablement (prerequisite for the model)

The system PyTorch is currently the **CPU-only** build
(`torch 2.6.0+cpu`, `cuda.is_available()==False`). Before real inference you must
install the **Jetson CUDA-enabled** PyTorch wheel matching JetPack 6.0 / CUDA
12.2, and `ultralytics`, **into the project venv**. This is a manual step — the
build deliberately does not modify JetPack/CUDA or auto-install wheels.

Manual commands to run yourself (verify the wheel URL against NVIDIA's current
Jetson PyTorch index for JetPack 6.0 / cu122 first):

```bash
cd ~/VytalLink
# 1) Install the Jetson CUDA torch wheel into the venv (URL per NVIDIA docs):
./.venv/bin/python -m pip install --no-cache-dir <jetson-cu122-torch-wheel-url>
# 2) Confirm CUDA is visible:
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 3) Install ultralytics:
./.venv/bin/python -m pip install ultralytics
```

> Note: because the venv was created with `--system-site-packages`, take care
> that the venv-installed torch shadows the system CPU build for the app.

## 4. Desired wearable

| # | Item | Notes |
| --- | --- | --- |
| 8 | Wearable choice | The device has not been selected. Pick one and provide its connectivity. |

To connect it, the following are needed: device model; connection method
(Bluetooth LE / vendor cloud API / local SDK); pairing/auth details (BLE MAC or
API key/secret — `.env` only); which metrics it exposes (HR, motion/accel,
battery, connection quality) and their units/ranges; and the sample rate. The
provider implements `vytallink.wearable.base.WearableProvider`
(`connect`/`read`/`disconnect`) behind the existing factory.

## 5. Desired notification method

| # | Item | Notes |
| --- | --- | --- |
| 9 | Notification channel(s) | Webhook is ready now (`WEBHOOK_URL` + `WEBHOOK_SECRET`, HMAC-SHA256 signed). For SMS/email/push, provide provider + credentials; each is a new `AlertProvider`. |

## 6. Hardware permissions / access

| # | Item | Notes |
| --- | --- | --- |
| 10 | Device permissions | `/dev/video0` is owned by `root:video`. To use a USB/CSI camera directly, ensure the run user is in the `video` group: `sudo usermod -aG video $USER` (then re-login). RTSP needs no local device permission. GPU/TensorRT may require the container/runtime device access if you later containerize. |

## 7. What stays deferred until the above is provided

- TensorRT engine export (only after the real model loads and ordinary GPU
  inference is confirmed).
- Any claim that real camera/model/wearable paths "work" — they are implemented
  and unit-tested for safe behavior, but marked **pending** until real hardware
  validation.
