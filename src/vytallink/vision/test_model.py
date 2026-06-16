"""Model inspection command — ``python -m vytallink.vision.test_model``.

Loads the configured fall model through the real :class:`YoloFallDetector` and
reports: sanitized model name, task, classes, the actual inference device, CUDA
availability, load success, warmup latency, and a synthetic inference latency.
All errors are sanitized. Returns a non-zero exit code on failure.

It never prints the model's absolute filesystem path or any credential.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vytallink.common.errors import DetectorError
from vytallink.config import PROJECT_ROOT, get_settings
from vytallink.monitoring import system_info
from vytallink.vision.detector_yolo import YoloFallDetector

DEFAULT_MODEL = PROJECT_ROOT / "models" / "fall_detection.pt"


def _resolve_model_path() -> str:
    settings = get_settings()
    if settings.model_path:
        return settings.model_path
    if DEFAULT_MODEL.exists():
        return str(DEFAULT_MODEL)
    return ""


def main() -> int:
    print("VytalLink model inspection")
    print("=" * 60)

    gpu = system_info.gpu_info()
    print(f"CUDA available : {gpu.get('available')}  ({gpu.get('detail')})")
    print(f"torch          : {gpu.get('torch_version')}  cuda_build={gpu.get('cuda_build')}")
    if gpu.get("device_name"):
        print(f"GPU device     : {gpu.get('device_name')}")

    model_path = _resolve_model_path()
    if not model_path:
        print(
            "\n[FAIL] No model found. Set MODEL_PATH in .env or place the model at "
            f"{DEFAULT_MODEL.relative_to(PROJECT_ROOT)} .",
            file=sys.stderr,
        )
        return 2

    safe_name = Path(model_path).name
    print(f"Model file     : {safe_name}")

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
    print(f"  warmup        : {detector.warmup_ms:.1f} ms" if detector.warmup_ms else "  warmup        : n/a")

    # Synthetic inference (no camera required).
    try:
        import numpy as np

        from vytallink.common.clock import SystemClock
        from vytallink.common.types import Frame

        img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
        frame = Frame(frame_id=1, timestamp=SystemClock().now(), source_id="test", width=640, height=480, image=img)
        dets = detector.infer(frame)
        print(f"  inference     : {detector.last_inference_ms:.1f} ms  (synthetic frame, {len(dets)} detections)")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] inference: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nRESULT: MODEL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
