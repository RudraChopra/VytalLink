"""Camera diagnostics — ``python -m vytallink.vision.test_camera``.

Connects to the configured RTSP camera and runs a stability test (default 60 s)
**without saving any footage or opening any window**. Reports resolution, frames
received, effective FPS, failed reads, reconnect count, stale-frame warnings, and
average read latency. Uses the provider's bounded-backoff reconnection, shuts
down cleanly, sanitizes every error, and returns a non-zero exit code on failure.

Run inference only *after* the camera is stable on its own.
"""

from __future__ import annotations

import argparse
import sys
import time

from vytallink.common.clock import SystemClock
from vytallink.config import get_settings
from vytallink.vision.rtsp import RTSPCamera


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VytalLink RTSP camera stability test")
    p.add_argument("--seconds", type=float, default=60.0, help="test duration (default 60)")
    p.add_argument("--fps", type=float, default=10.0, help="read poll rate (default 10)")
    p.add_argument("--device-id", default=None, help="override camera device id label")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    url = settings.rtsp_url()

    print("VytalLink camera diagnostics")
    print("=" * 60)
    if not url:
        print(
            "[FAIL] No RTSP target configured. Set CAMERA_HOST (+ CAMERA_PORT / "
            "CAMERA_STREAM_PATH) or a full CAMERA_SOURCE URL in .env.",
            file=sys.stderr,
        )
        return 2

    clock = SystemClock()
    cam = RTSPCamera(
        url,
        source_id=args.device_id or settings.camera_device_id,
        clock=clock,
        stale_timeout=max(2.0, 3.0 / max(args.fps, 0.1)) if args.fps else 5.0,
    )
    print(f"Source     : {cam.safe_source}")   # credentials redacted
    print(f"Duration   : {args.seconds:.0f}s   poll: {args.fps:.0f} Hz")
    print("-" * 60)

    frames = 0
    failed = 0
    stale_warnings = 0
    latencies: list[float] = []
    interval = 1.0 / max(args.fps, 0.1)
    start = time.monotonic()
    next_report = start + 5.0

    try:
        while (time.monotonic() - start) < args.seconds:
            t0 = time.perf_counter()
            frame = cam.read()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if frame is not None:
                frames += 1
            else:
                failed += 1
                if cam.is_stale():
                    stale_warnings += 1
            now = time.monotonic()
            if now >= next_report:
                h = cam.health()
                print(
                    f"  t={now - start:5.1f}s  frames={frames:5d}  fps={h.get('effective_fps')}  "
                    f"failed={failed}  reconnects={h.get('reconnects')}  "
                    f"dropped={h.get('frames_dropped')}  status={h.get('status')}"
                )
                next_report = now + 5.0
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        cam.close()

    h = cam.health()
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None
    print("-" * 60)
    print(f"resolution        : {h.get('resolution')}")
    print(f"frames received   : {frames}")
    print(f"effective fps     : {h.get('effective_fps')}")
    print(f"failed reads      : {failed}")
    print(f"reconnects        : {h.get('reconnects')}")
    print(f"stale warnings    : {stale_warnings}")
    print(f"frames dropped    : {h.get('frames_dropped')}  (intentional: latest-frame)")
    print(f"avg read latency  : {avg_latency} ms")
    if h.get("last_error"):
        print(f"last error        : {h.get('last_error')}")  # already sanitized upstream

    if frames == 0:
        print("\nRESULT: CAMERA_FAIL (no frames received — check network/credentials/path)", file=sys.stderr)
        return 1
    print("\nRESULT: CAMERA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
