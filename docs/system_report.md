# VytalLink — System Report

Generated during the Phase 1 overnight build by safe, read-only inspection of
the host. No secrets, credentials, or credential-bearing URLs are included.

**Host inspected:** NVIDIA Jetson Orin Nano Developer Kit
**Inspection date:** 2026-06-15 (UTC build session)

---

## 1. Platform

| Item | Value |
| --- | --- |
| Jetson model | **NVIDIA Jetson Orin Nano Developer Kit** |
| CPU architecture | `aarch64` (ARM 64-bit) |
| CPU cores | 6 |
| Ubuntu version | **22.04.5 LTS** (Jammy Jellyfish) |
| Linux kernel | `5.15.136-tegra` |
| L4T release | **R36.3.0** (GCID 36923193, 2024-07-19) |
| JetPack version | **6.0 GA** (corresponds to L4T R36.3) |

> Note: the `nvidia-jetpack` apt metapackage was not listed, but L4T R36.3.0
> maps to JetPack 6.0. The individual JetPack components (CUDA, cuDNN, TensorRT)
> are installed and detected below.

## 2. Toolchain / runtime

| Item | Value | Notes |
| --- | --- | --- |
| Python | **3.10.12** (`/usr/bin/python3`) | system interpreter |
| pip | 25.0.1 | from `~/.local` |
| `venv` module | available | used for project environment |
| CUDA | **12.2** (`nvcc V12.2.140`, `/usr/local/cuda-12.2`) | toolkit present |
| TensorRT | **8.6.2.3** (`+cuda12.2`); python `tensorrt 8.6.2` | system bindings present |
| PyTorch | **2.6.0+cpu** | ⚠️ CPU-only wheel installed |
| PyTorch CUDA access | **False** | ⚠️ see note below |
| OpenCV | **4.11.0** (`cv2`) | importable from system Python |
| Docker | **29.5.3** | available |
| NVIDIA container runtime | **1.14.2** (`nvidia-container-toolkit`) | available |
| FFmpeg | **4.4.2** | available |
| GStreamer | **1.20.3** (`gst-launch-1.0`) | available |

> ⚠️ **PyTorch is the CPU-only build** (`2.6.0+cpu`) and therefore reports
> `torch.cuda.is_available() == False`. This does **not** affect Phase 1, which
> runs entirely in simulation mode and does not perform GPU inference. Before
> the real YOLO fall model is used, the Jetson-specific CUDA-enabled PyTorch
> wheel (matching JetPack 6.0 / CUDA 12.2) must be installed. The exact manual
> command is documented in `docs/hardware_needed.md` and
> `docs/environment_decision.md`. We deliberately do **not** auto-install or
> replace PyTorch tonight, per the operating boundaries.

## 3. Resources

| Item | Value |
| --- | --- |
| RAM | 7.4 GiB total, ~5.3 GiB available |
| Swap | 3.7 GiB |
| Disk (`/`) | 233 GB total, **192 GB free** (13% used) |

## 4. Network interfaces (no addresses treated as secret, LAN only)

| Interface | State | Address (LAN) |
| --- | --- | --- |
| `wlan0` | UP | 192.168.86.29/24 |
| `eth0` | UP | 192.168.42.3/24 |
| `docker0` | DOWN | 172.17.0.1/16 |
| `lo` | UP | 127.0.0.1/8 |
| `can0`, `usb0`, `usb1`, `l4tbr0` | DOWN | — |

The dashboard will bind to `0.0.0.0:5050` by default, reachable on the LAN at
`http://192.168.86.29:5050` (Wi-Fi) and `http://192.168.42.3:5050` (Ethernet).

## 5. Video devices

| Device | Notes |
| --- | --- |
| `/dev/video0` | Present (owned by `root:video`). A capture device is attached. Phase 1 does **not** open it by default; the default `VISION_MODE=simulation` is used. No credentials were probed. |

## 6. Project directory & Git

- Working directory: `/home/rchopra/VytalLink` — was **empty** at start.
- Git: **not initialized** at start. Initialized on `main` during this build.

---

## Summary of implications for Phase 1

1. **Simulation-first is the correct default.** All hardware-dependent paths
   (RTSP camera, GPU model inference, real wearable) are implemented as clean
   adapters but left dormant until real configuration / weights are supplied.
2. **A `--system-site-packages` venv** is the reliable environment: it preserves
   access to the system-tuned `cv2`, `tensorrt`, and the eventual CUDA PyTorch,
   while isolating the small set of pure-Python web/test dependencies.
3. **PyTorch must be upgraded to a CUDA Jetson wheel before real inference** —
   tracked as a manual hardware-enablement step, not done automatically.
4. Ample disk and RAM are available for Phase 1.
