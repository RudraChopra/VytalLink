"""Simultaneous two-camera diagnostic — ``python -m vytallink.vision.multi_camera_test``.

Runs every enabled ``CAMERA_{N}_*`` camera concurrently through the real shared
pipeline (one loaded model + one inference lane + per-camera workers) for a fixed
duration (default 60 s) and reports per-camera and combined metrics. It saves no
video/images, redacts all credentials/URLs (only credential-free camera ids and
host:port labels are printed), and never starts the production web server.

Configure cameras in ``.env`` (or the process env), e.g.::

    CAMERA_1_ENABLED=true CAMERA_1_HOST=... CAMERA_1_USERNAME=... CAMERA_1_PASSWORD=... CAMERA_1_STREAM_PATH=/stream1
    CAMERA_2_ENABLED=true CAMERA_2_HOST=... ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from vytallink.common.clock import SystemClock
from vytallink.config import get_settings
from vytallink.monitoring import system_info
from vytallink.vision.multi_camera import build_multi_camera_monitor

try:  # psutil ships with the project deps
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _mps_allocated_mb() -> float | None:
    try:
        import torch  # noqa: WPS433

        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "current_allocated_memory"):
            return round(mps.current_allocated_memory() / 1e6, 1)
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VytalLink simultaneous multi-camera diagnostic")
    p.add_argument("--seconds", type=float, default=60.0, help="duration (default 60)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    cameras = settings.configured_cameras()

    print("VytalLink simultaneous multi-camera diagnostic")
    print("=" * 64)
    if len(cameras) < 1:
        print("[FAIL] No enabled CAMERA_{N}_* cameras configured. See docs/rtsp_camera_test.md.",
              file=sys.stderr)
        return 2

    gpu = system_info.gpu_info()
    print(f"device   : {gpu.get('selected_device')}  (mps={gpu.get('mps_available')})")
    print(f"model    : {os.path.basename(settings.model_path) if settings.model_path else 'fall_detection.pt (default)'}")
    print(f"cameras  : {', '.join(c.safe_label() for c in cameras)}")   # host:port, no creds
    print(f"duration : {args.seconds:.0f}s   detect_max_fps={settings.detect_max_fps}/cam")
    print("-" * 64)

    monitor = build_multi_camera_monitor(settings, cameras, clock=SystemClock())
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc is not None:
        proc.cpu_percent(None)  # prime
    mem_start_mb = round(proc.memory_info().rss / 1e6, 1) if proc else None

    monitor.start()
    cpu_samples: list[float] = []
    mem_samples: list[float] = []
    start = time.monotonic()
    next_report = start + 5.0
    try:
        while (time.monotonic() - start) < args.seconds:
            time.sleep(0.5)
            now = time.monotonic()
            if now >= next_report:
                if proc is not None:
                    cpu_samples.append(proc.cpu_percent(None))
                    mem_samples.append(round(proc.memory_info().rss / 1e6, 1))
                line = []
                for w in monitor.workers:
                    h = w.health()
                    line.append(f"{w.camera_id}: fps={h['fps']} infms={h['infer_ms_avg']} "
                                f"recon={h['reconnects']} falls={h['confirmed_falls']} "
                                f"backlog={h['backlog']} {'STALE' if h['stale'] else 'ok'}")
                print(f"  t={now - start:5.1f}s  qdepth={monitor.queue_depth}  " + " | ".join(line))
                next_report = now + 5.0
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        elapsed = monitor.elapsed
        snapshots = {w.camera_id: w.metrics(elapsed) for w in monitor.workers}
        peak_q = monitor.peak_queue_depth
        load_count = monitor.model_load_count
        results = monitor.stop()

    mem_end_mb = round(proc.memory_info().rss / 1e6, 1) if proc else None

    # ---- per-camera report -------------------------------------------------
    total_infer = 0.0
    any_delayed = False
    print("-" * 64)
    for cam_id, m in snapshots.items():
        delayed = bool(m["stale"]) or (m.get("last_frame_age_ms") or 0) > 1000
        any_delayed = any_delayed or delayed
        total_infer += m.get("inference_fps", 0.0) or 0.0
        print(f"[{cam_id}]")
        print(f"  connection      : {'ok' if m['connected'] else 'down'} (status={m['status']})")
        print(f"  resolution      : {m['resolution']}")
        print(f"  capture FPS     : {m['fps']}  (unique frames/s)")
        print(f"  inference FPS   : {m['inference_fps']}")
        print(f"  end-to-end FPS  : {m['end_to_end_fps']}")
        print(f"  read latency    : avg {m['read_ms_avg']} ms  p95 {m['read_ms_p95']} ms")
        print(f"  infer latency   : avg {m['infer_ms_avg']} ms  p95 {m['infer_ms_p95']} ms")
        print(f"  dropped frames  : {m['dropped_frames']}   failed reads: {m['failed_reads']}")
        print(f"  reconnects      : {m['reconnects']}   backlog: {m['backlog']}")
        print(f"  detected classes: {m['detected_classes'] or 'none'}")
        print(f"  confirmed falls : {m['confirmed_falls']}")
        print(f"  delayed?        : {'YES' if delayed else 'no'}")

    # ---- combined report ---------------------------------------------------
    print("-" * 64)
    print("[combined]")
    print(f"  model load count : {load_count}  (shared across {len(snapshots)} cameras)")
    print(f"  total inference  : {round(total_infer, 2)} fps")
    print(f"  inference qdepth : peak {peak_q} (<= #cameras => bounded, no backlog growth)")
    print(f"  CPU usage (proc) : avg {round(sum(cpu_samples)/len(cpu_samples), 1) if cpu_samples else 'n/a'}%")
    print(f"  memory (proc)    : start {mem_start_mb} MB -> end {mem_end_mb} MB")
    print(f"  MPS allocated    : {_mps_allocated_mb()} MB" if _mps_allocated_mb() is not None else "  MPS allocated    : n/a")
    print(f"  either delayed?  : {'YES' if any_delayed else 'no'}")
    print(f"  clean shutdown   : {results}")

    runs_created = os.path.isdir("runs")
    print(f"  runs/ created?   : {'YES (unexpected)' if runs_created else 'no'}")

    ok = all(results.values()) and all(m["connected"] for m in snapshots.values()) and not runs_created
    print("\nRESULT: " + ("MULTI_OK" if ok else "MULTI_DEGRADED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
