"""Ultralytics YOLO fall-detector adapter (dormant in Phase 1).

This adapter is implemented cleanly but is NOT activated until a real
``MODEL_PATH`` is supplied and ordinary GPU inference is confirmed. It does not
download any weights and does not pretend to be the VytalLink fall model.

Requirements to enable (see docs/hardware_needed.md):
  * The Jetson CUDA-enabled PyTorch wheel installed (system torch is CPU-only).
  * ``pip install ultralytics`` in the project venv.
  * A trained fall model at MODEL_PATH whose class names include a fall class
    (configurable via FALL_CLASS_NAMES).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vytallink.common.errors import DetectorError
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus, RawDetection
from vytallink.vision.detector_base import FallDetector

log = get_logger("vision.detector.yolo")


class YoloFallDetector(FallDetector):
    name = "yolo"

    def __init__(self, model_path: str, *, image_size: int = 416, device: str | None = None):
        self.model_path = model_path
        self.image_size = image_size
        self.device = device
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not self.model_path or not Path(self.model_path).expanduser().exists():
            raise DetectorError(
                f"YOLO model not found at MODEL_PATH={self.model_path!r}. "
                "Provide a trained fall model. See docs/hardware_needed.md."
            )
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy, optional dep)
        except ImportError as exc:
            raise DetectorError(
                "ultralytics is not installed in the venv. Install it (and the "
                "Jetson CUDA PyTorch wheel) before enabling DETECTOR_MODE=yolo. "
                "See docs/hardware_needed.md."
            ) from exc
        try:
            self._model = YOLO(str(Path(self.model_path).expanduser()))
            if self.device:
                self._model.to(self.device)
        except Exception as exc:  # pragma: no cover - requires real weights
            raise DetectorError(f"Failed to load YOLO model: {exc}") from exc
        log.info("YOLO model loaded from %s", self.model_path)

    def infer(self, frame: Frame) -> list[RawDetection]:
        if self._model is None:
            raise DetectorError("YOLO model not loaded; call load() first")
        if frame.image is None:  # pragma: no cover - requires real frames
            return []
        results = self._model.predict(  # pragma: no cover - requires real weights
            frame.image, imgsz=self.image_size, verbose=False
        )
        detections: list[RawDetection] = []
        for result in results:  # pragma: no cover - requires real weights
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0]]
                detections.append(
                    RawDetection(
                        timestamp=frame.timestamp,
                        class_id=cls_id,
                        class_name=str(names.get(cls_id, str(cls_id))),
                        confidence=conf,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        source_id=frame.source_id,
                        frame_id=frame.frame_id,
                        metadata={"simulated": False},
                    )
                )
        return detections

    def close(self) -> None:
        self._model = None

    def health(self) -> dict[str, Any]:
        return {
            "status": (HealthStatus.OK if self.loaded else HealthStatus.DOWN).value,
            "name": self.name,
            "loaded": self.loaded,
            "model_path": self.model_path,
            "simulated": False,
        }
