"""Discriminating mechanism test for the Apple-silicon MPS startup abort.

Hypothesis: the Metal "addScheduledHandler after commit" abort is caused by an
MPS tensor op running on a NON-inference thread (the loop thread, via
`/health` -> gpu_info() -> device_report() -> select_device() -> _probe_mps())
concurrently with steady-state inference on the dedicated inference thread.

This reproduces it in-process in seconds, with no cameras and no HTTP server:

  thread A: detector.infer(blank) in a tight loop   (the inference lane)
  thread B: system_info.gpu_info() in a tight loop  (the /health probe path)

`gpu_info()` is cached, so to model the *first* /health probe racing inference
we clear the device cache each iteration, forcing the probe to re-run on thread
B. If the mechanism is real, the process aborts (signal 6 / "Abort trap: 6").
After the fix, gpu_info()/device_report() must never create an MPS tensor, so
thread B does no Metal work and the process survives.

Exit code: 0 = survived the window, nonzero/-6 = aborted (race reproduced).
Saves nothing; touches no credentials. Run via diagnostics/run_mps_race.sh.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from vytallink.common.types import Frame
from vytallink.config import get_settings
from vytallink.monitoring import system_info
from vytallink.common.device import reset_device_cache
from vytallink.vision.factory import build_detector


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    settings = get_settings()
    detector = build_detector(settings)
    detector.load()  # real MPS load + warmup on the main thread (sequential, safe)
    device = detector.health().get("device")
    if device != "mps":
        print("SKIP: detector not on MPS (device=%s)" % device)
        return 0

    frame = Frame(frame_id=1, timestamp=settings_now(), source_id="probe",
                  width=640, height=480, image=np.zeros((480, 640, 3), dtype="uint8"))
    stop = threading.Event()
    errors: list[str] = []

    def infer_loop():
        while not stop.is_set():
            try:
                detector.infer(frame)
            except Exception as exc:  # python-level errors are NOT the abort we hunt
                errors.append("infer:%s" % type(exc).__name__)
                return

    def probe_loop():
        # Model the /health readiness probe re-running the MPS device probe on a
        # different thread than inference.
        while not stop.is_set():
            try:
                reset_device_cache()
                system_info.gpu_info()
            except Exception as exc:
                errors.append("probe:%s" % type(exc).__name__)
                return

    a = threading.Thread(target=infer_loop, name="infer-A", daemon=True)
    b = threading.Thread(target=probe_loop, name="probe-B", daemon=True)
    a.start(); b.start()
    time.sleep(duration)
    stop.set()
    a.join(timeout=2.0); b.join(timeout=2.0)
    print("SURVIVED window=%.1fs python_errors=%s" % (duration, errors or "none"))
    return 0


def settings_now():
    from vytallink.common.clock import SystemClock
    return SystemClock().now()


if __name__ == "__main__":
    raise SystemExit(main())
