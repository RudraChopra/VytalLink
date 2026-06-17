"""Model inspection + benchmark — ``python -m vytallink.vision.test_model``.

Loads the configured fall model through the real :class:`YoloFallDetector` and
reports, with everything sanitized (never an absolute path or credential):

* host: OS, architecture, Python, Torch, CUDA/MPS availability, selected device,
* model: sanitized name, task, class names, and the expected fall classes,
* a synthetic inference (no camera needed), then
* a benchmark: one warmup inference plus >= 10 measured steady-state
  inferences, reporting image size, warmup / average / median / min / max
  latency and approximate FPS. The accelerator is synchronized before timing.

If an op is unsupported on the selected accelerator (e.g. MPS), the detector
falls back to CPU and this command reports the fallback rather than pretending
the accelerator was used. Returns a non-zero exit code on failure.

This is a standalone validation tool: it does NOT change the running app, which
stays configured for simulation.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

from vytallink.common.errors import DetectorError
from vytallink.config import PROJECT_ROOT, get_settings
from vytallink.monitoring import system_info
from vytallink.vision.detector_yolo import YoloFallDetector

DEFAULT_MODEL = PROJECT_ROOT / "models" / "fall_detection.pt"
EXPECTED_CLASSES = {"fallen", "sitting", "standing"}
STEADY_STATE_ITERS = 15  # >= 10 measured inferences


def _resolve_model_path() -> str:
    settings = get_settings()
    if settings.model_path:
        return settings.model_path
    if DEFAULT_MODEL.exists():
        return str(DEFAULT_MODEL)
    return ""


def _print_host() -> None:
    gpu = system_info.gpu_info()
    py = sys.version_info
    print(f"OS / arch      : {platform.system()} {platform.machine()}")
    print(f"Python         : {py.major}.{py.minor}.{py.micro}")
    print(f"torch          : {gpu.get('torch_version')}  cuda_build={gpu.get('cuda_build')}")
    print(f"CUDA available : {gpu.get('cuda_available', gpu.get('available'))}")
    print(f"MPS available  : {gpu.get('mps_available')}  (built={gpu.get('mps_built')})")
    print(f"Selected device: {gpu.get('selected_device')}")
    if gpu.get("device_name"):
        print(f"GPU device     : {gpu.get('device_name')}")


def _bench(detector: YoloFallDetector) -> dict[str, float]:
    """Run one warmup + STEADY_STATE_ITERS measured inferences on a synthetic
    image, synchronizing the accelerator before reading each timer."""
    import numpy as np  # noqa: WPS433

    img = (np.random.rand(detector.image_size, detector.image_size, 3) * 255).astype("uint8")

    # Warmup (not measured into steady-state stats).
    t0 = time.perf_counter()
    detector._predict(img)
    detector._device_sync()
    warmup_ms = (time.perf_counter() - t0) * 1000.0

    samples: list[float] = []
    for _ in range(STEADY_STATE_ITERS):
        t = time.perf_counter()
        detector._predict(img)
        detector._device_sync()  # ensure GPU/MPS work completed before timing
        samples.append((time.perf_counter() - t) * 1000.0)

    avg = statistics.mean(samples)
    return {
        "warmup_ms": warmup_ms,
        "avg_ms": avg,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "fps": (1000.0 / avg) if avg > 0 else 0.0,
        "iters": float(len(samples)),
    }


def main() -> int:
    print("VytalLink model inspection + benchmark")
    print("=" * 60)
    _print_host()

    model_path = _resolve_model_path()
    if not model_path:
        print(
            "\n[FAIL] No model found. Set MODEL_PATH in .env or place the model at "
            f"{DEFAULT_MODEL.relative_to(PROJECT_ROOT)} .",
            file=sys.stderr,
        )
        return 2

    print(f"Model file     : {Path(model_path).name}")

    settings = get_settings()
    detector = YoloFallDetector(
        model_path,
        image_size=settings.image_size,
        confidence=settings.confidence_threshold,
        require_transition=settings.require_fall_transition,
    )

    try:
        detector.load()
    except DetectorError as exc:
        print(f"\n[FAIL] load: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - sanitize anything unexpected
        print(f"\n[FAIL] load: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n[OK] model loaded")
    print(f"  task          : {detector.task}")
    print(f"  classes       : {detector.class_names}")
    print(f"  device        : {detector.device_str}  (half={detector.half})")

    # Validate the expected fall classes are present.
    present = {str(v).lower() for v in detector.class_names.values()}
    missing = EXPECTED_CLASSES - present
    if missing:
        print(f"\n[FAIL] model is missing expected classes: {sorted(missing)} "
              f"(have {sorted(present)})", file=sys.stderr)
        return 1
    print(f"  expected cls  : {sorted(EXPECTED_CLASSES)} all present  ✓")

    # Benchmark (warmup + measured steady-state).
    try:
        stats = _bench(detector)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] benchmark: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if detector.mps_fallback_reason:
        print(f"\n[WARN] accelerator fell back to CPU: {detector.mps_fallback_reason}")
        print(f"       inference device is now: {detector.device_str}")

    print("\nBenchmark (synthetic frames)")
    print(f"  image size    : {detector.image_size}x{detector.image_size}")
    print(f"  device        : {detector.device_str}")
    print(f"  warmup        : {stats['warmup_ms']:.1f} ms")
    print(f"  measured iters: {int(stats['iters'])}")
    print(f"  average       : {stats['avg_ms']:.1f} ms")
    print(f"  median        : {stats['median_ms']:.1f} ms")
    print(f"  min / max     : {stats['min_ms']:.1f} ms / {stats['max_ms']:.1f} ms")
    print(f"  approx FPS    : {stats['fps']:.1f}")

    print("\nRESULT: MODEL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
