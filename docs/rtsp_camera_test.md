# One-camera RTSP validation (Mac → Tapo, direct)

Validate a single Tapo camera over **direct RTSP** from the Mac, then run the
fall model on the live stream using Apple **MPS**. This reuses the production
camera pipeline (`RTSPCamera` → `YoloFallDetector` → fall state machine); it is
**not** a separate test system. No footage is saved and the RTSP URL/credentials
are always redacted.

> RTSP stays **disabled in committed defaults** (`.env.example` is
> `VISION_MODE=simulation`). Real camera credentials live **only in your local
> `.env`**, which is git-ignored. Do not commit `.env`.

## Prerequisites

1. **Network reachability.** The Mac must be able to reach the Tapo directly on
   its IP:554. The Mac is currently on `192.168.42.x`. If the Tapo lives on the
   Jetson's private wired network (not the Mac's subnet), direct RTSP will not
   work from the Mac — use the existing `http_mjpeg` Jetson relay instead. Check:
   ```bash
   nc -vz -w 4 <TAPO_IP> 554     # "succeeded" => reachable
   ```
2. **Tapo Camera Account.** RTSP uses the camera's local *Camera Account*
   (Tapo app → Camera Settings → Advanced → Camera Account), **not** your
   TP-Link cloud login. Tapo stream paths: `stream1` (HD) or `stream2` (lower-res).

## Exact `.env` variables for one Tapo RTSP camera

Edit your local `.env` (never `.env.example`). Set these and leave the
`CAMERA_HTTP_*` relay fields blank:

```ini
VISION_MODE=rtsp
DETECTOR_MODE=yolo
MODEL_PATH=models/fall_detection.pt

# Component form (recommended for Tapo). Credentials stay separate so they are
# never part of a logged/echoed source string.
CAMERA_HOST=<TAPO_IP_ON_MACS_SUBNET>     # e.g. 192.168.42.71
CAMERA_PORT=554
CAMERA_STREAM_PATH=stream1               # or stream2 for the sub-stream
CAMERA_USERNAME=<tapo_camera_account_user>
CAMERA_PASSWORD=<tapo_camera_account_password>

# Leave the full-URL and relay forms empty so the component form is used:
CAMERA_SOURCE=
CAMERA_HTTP_STREAM_URL=
CAMERA_HTTP_SNAPSHOT_URL=
CAMERA_HTTP_BEARER_TOKEN=
```

Equivalent single-line form (instead of the component fields):
`CAMERA_SOURCE=rtsp://<user>:<pass>@<TAPO_IP>:554/stream1` — but the component
form keeps the password out of one combined field. Either way the assembled URL
is only ever surfaced as `rtsp://***REDACTED***@host:port/path`.

## Run the tests (manual — the full app stays off)

**1) Connectivity / stability test (30 s, no inference, no saving):**

```bash
./.venv/bin/python -m vytallink.vision.test_camera --seconds 30
```
Reports: redacted source, resolution, effective FPS, failed reads, reconnects,
stale warnings, dropped frames, average read latency. Expect `RESULT: CAMERA_OK`.

**2) Live fall detection on the real stream using MPS (one camera):**

```bash
./.venv/bin/python -m vytallink.vision.live_detection --seconds 30
```
Forces `VISION_MODE=rtsp` + `DETECTOR_MODE=yolo` for the run (using your `.env`
camera/model/threshold values), loads the model once on the selected device
(MPS on Apple silicon), and reports: camera connection status, selected device,
resolution, capture FPS, inference FPS, end-to-end FPS, average inference
latency, dropped frames, reconnects, any device fallback, and detected class
counts (`fallen`/`sitting`/`standing`). It does **not** send caregiver alerts
unless `--alerts` is passed, and never saves footage or opens a window. Expect
`RESULT: LIVE_OK`.

## Safety / privacy

- No video or images are written to disk by either command.
- The RTSP URL and credentials are redacted in all logs, health, and terminal
  output (`rtsp://***REDACTED***@host:port/path`).
- The full production server is **not** started by these commands.
- After validating one camera, stop here — do not enable a second camera or
  start the full app until the single-camera results are reviewed.

## Revert to safe defaults / the Jetson relay

To go back to simulation, set `VISION_MODE=simulation` (and `DETECTOR_MODE=
simulation`) in `.env`. To use the Jetson relay from the Mac instead of direct
RTSP, set `VISION_MODE=http_mjpeg` with the `CAMERA_HTTP_*` fields (see
`.env.example`). The committed defaults already keep RTSP disabled.
