"""Ultralytics YOLO fall-detector adapter — the real VytalLink fall detector.

Runs the legacy VytalLink v1 model ``fall_detection.pt`` (a custom **YOLO11n**
detection model, classes ``0=fallen, 1=sitting, 2=standing``). A fall is class 0
``fallen``. This adapter:

* loads the model **once** (never per frame) and resolves the real device
  (``cuda:0`` on the Jetson once the CUDA wheel is installed, else ``cpu``),
* warms the model up and records warmup + per-frame inference latency,
* maps each detection box to a :class:`RawDetection` with its posture class name,
* optionally applies the legacy "DTS-lite" upright→fallen :class:`PostureTransitionGate`
  so a person who is *already lying down* does not register as a fall (a bare
  ``fallen`` posture is surfaced as the non-alerting ``fallen_posture`` class
  until a genuine transition is seen),
* never logs or returns the model's absolute filesystem path.

It does not download weights and does not pretend a pose model has a fall class.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from vytallink.common.clock import Clock, SystemClock
from vytallink.common.errors import DetectorError
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus, RawDetection
from vytallink.vision.detector_base import FallDetector
from vytallink.vision.posture_gate import GateConfig, PostureTransitionGate

log = get_logger("vision.detector.yolo")

#: Class name used for a ``fallen`` posture that is NOT (yet) a confirmed fall
#: transition. It is deliberately NOT a fall-class name, so it is not evidence.
FALLEN_POSTURE_CLASS = "fallen_posture"


class YoloFallDetector(FallDetector):
    name = "yolo"

    def __init__(
        self,
        model_path: str,
        *,
        image_size: int = 416,
        confidence: float = 0.55,
        device: str | None = None,
        half: bool | None = None,
        require_transition: bool = True,
        gate_config: GateConfig | None = None,
        clock: Clock | None = None,
        warmup: bool = True,
    ) -> None:
        self.model_path = model_path
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self._device_pref = device
        self._half_pref = half
        self.require_transition = require_transition
        self.clock: Clock = clock or SystemClock()
        self._warmup = warmup
        self._gate = PostureTransitionGate(gate_config, clock=self.clock)

        self._model: Any = None
        self.device_str: str = "cpu"
        self.half: bool = False
        self.task: str | None = None
        self.class_names: dict[int, str] = {}
        self.warmup_ms: float | None = None

        # metrics
        self.inference_count = 0
        self.last_inference_ms: float | None = None
        self.avg_inference_ms: float | None = None
        self._infer_marks: deque[float] = deque(maxlen=30)
        self._last_error: str | None = None
        #: Whether the most recent inference call succeeded. Drives runtime health
        #: so a model that loaded once but fails every frame is reported DEGRADED.
        self.last_infer_ok: bool = True

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> tuple[str, bool]:
        if self._device_pref:
            dev = self._device_pref
            cuda = dev.startswith("cuda")
        else:
            try:
                import torch  # noqa: WPS433

                cuda = bool(torch.cuda.is_available())
            except Exception:  # pragma: no cover - torch always present here
                cuda = False
            dev = "cuda:0" if cuda else "cpu"
        # NOTE: fp16 is OFF by default. On this Jetson (cuDNN 8.6 + torch 2.3)
        # several YOLO11 conv plans raise CUDNN_STATUS_NOT_SUPPORTED in half
        # precision and fall back to a slow path (~700 ms/frame vs ~35 ms fp32).
        # Enable explicitly (DETECTOR_HALF) only if a future cuDNN supports it.
        half = bool(self._half_pref) and cuda
        return dev, half

    def load(self) -> None:
        expanded = Path(self.model_path).expanduser() if self.model_path else None
        if expanded is None or not expanded.exists():
            raise DetectorError(
                f"YOLO model not found at MODEL_PATH={self._safe_model!r}. "
                "Provide the trained fall model. See docs/hardware_needed.md."
            )
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy, optional dep)
        except ImportError as exc:
            raise DetectorError(
                "ultralytics is not installed in the venv. Install it (and the "
                "Jetson CUDA PyTorch wheel) before enabling DETECTOR_MODE=yolo. "
                "See docs/hardware_needed.md."
            ) from exc

        self.device_str, self.half = self._resolve_device()
        try:
            self._model = YOLO(str(expanded))
            self._model.to(self.device_str)
            self.task = getattr(self._model, "task", None)
            self.class_names = dict(getattr(self._model.model, "names", {}) or {})
        except Exception as exc:
            self._model = None
            raise DetectorError(f"Failed to load YOLO model: {type(exc).__name__}: {exc}") from exc

        if self._warmup:
            try:
                self.warmup_ms = self._run_warmup()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("YOLO warmup failed (continuing): %s", type(exc).__name__)

        log.info(
            "YOLO model '%s' loaded: task=%s classes=%s device=%s half=%s warmup=%sms",
            self._safe_model,
            self.task,
            self.class_names,
            self.device_str,
            self.half,
            round(self.warmup_ms, 1) if self.warmup_ms is not None else "n/a",
        )

    def _run_warmup(self) -> float:
        import numpy as np  # noqa: WPS433

        blank = np.zeros((self.image_size, self.image_size, 3), dtype="uint8")
        t0 = time.perf_counter()
        self._predict(blank)
        self._cuda_sync()
        return (time.perf_counter() - t0) * 1000.0

    def _cuda_sync(self) -> None:
        if self.device_str.startswith("cuda"):
            try:
                import torch  # noqa: WPS433

                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - defensive
                pass

    def _predict(self, image: Any) -> Any:
        return self._model.predict(
            image,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device_str,
            half=self.half,
            verbose=False,
        )

    # -- inference ---------------------------------------------------------
    def infer(self, frame: Frame) -> list[RawDetection]:
        if self._model is None:
            raise DetectorError("YOLO model not loaded; call load() first")
        if frame.image is None:
            # No pixel data (e.g. a synthesized simulation frame) — nothing to do.
            return []

        t0 = time.perf_counter()
        try:
            results = self._predict(frame.image)
            self._cuda_sync()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.last_infer_ok = False
            log.warning("YOLO inference failed: %s", self._last_error)
            return []
        self._record_latency((time.perf_counter() - t0) * 1000.0)
        self.last_infer_ok = True
        self._last_error = None

        raw_boxes = self._extract_boxes(results)
        best_fallen, best_upright, has_det = self._summarize(raw_boxes)

        # Gate decides whether a fallen posture is a genuine fall *event*.
        if self.require_transition:
            fall_is_event = self._gate.observe(
                fallen_conf=best_fallen,
                upright_conf=best_upright,
                has_detection=has_det,
            )
        else:
            fall_is_event = best_fallen >= self.confidence

        return self._build_detections(frame, raw_boxes, fall_is_event)

    def _extract_boxes(self, results: Any) -> list[tuple[int, str, float, tuple[float, float, float, float]]]:
        out: list[tuple[int, str, float, tuple[float, float, float, float]]] = []
        for result in results:
            names = getattr(result, "names", {}) or self.class_names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0]]
                name = str(names.get(cls_id, str(cls_id))).lower()
                out.append((cls_id, name, conf, (xyxy[0], xyxy[1], xyxy[2], xyxy[3])))
        return out

    @staticmethod
    def _summarize(boxes: list[tuple[int, str, float, tuple]]) -> tuple[float, float, bool]:
        best_fallen = 0.0
        best_upright = 0.0
        for _cls, name, conf, _bbox in boxes:
            if name == "fallen":
                best_fallen = max(best_fallen, conf)
            elif name in ("sitting", "standing"):
                best_upright = max(best_upright, conf)
        return best_fallen, best_upright, bool(boxes)

    def _build_detections(
        self,
        frame: Frame,
        boxes: list[tuple[int, str, float, tuple]],
        fall_is_event: bool,
    ) -> list[RawDetection]:
        detections: list[RawDetection] = []
        for cls_id, name, conf, bbox in boxes:
            class_name = name
            if name == "fallen" and self.require_transition and not fall_is_event:
                # A fallen posture that is not (yet) a confirmed transition.
                # Surfaced for visibility but NOT as fall evidence.
                class_name = FALLEN_POSTURE_CLASS
            detections.append(
                RawDetection(
                    timestamp=frame.timestamp,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    source_id=frame.source_id,
                    frame_id=frame.frame_id,
                    metadata={"simulated": False, "raw_class": name},
                )
            )
        return detections

    def _record_latency(self, ms: float) -> None:
        self.inference_count += 1
        self.last_inference_ms = round(ms, 2)
        if self.avg_inference_ms is None:
            self.avg_inference_ms = ms
        else:
            self.avg_inference_ms = 0.8 * self.avg_inference_ms + 0.2 * ms
        self._infer_marks.append(self.clock.monotonic())

    def inference_fps(self) -> float:
        if len(self._infer_marks) < 2:
            return 0.0
        span = self._infer_marks[-1] - self._infer_marks[0]
        if span <= 0:
            return 0.0
        return round((len(self._infer_marks) - 1) / span, 2)

    def close(self) -> None:
        self._model = None

    # -- health ------------------------------------------------------------
    @property
    def _safe_model(self) -> str:
        """The model *basename* only — never the absolute filesystem path."""
        if not self.model_path:
            return "(unset)"
        return Path(self.model_path).name

    def _runtime_status(self) -> HealthStatus:
        if not self.loaded:
            return HealthStatus.DOWN
        # Loaded but the most recent inference raised → functionally degraded.
        if not self.last_infer_ok:
            return HealthStatus.DEGRADED
        return HealthStatus.OK

    def health(self) -> dict[str, Any]:
        return {
            "status": self._runtime_status().value,
            "name": self.name,
            "loaded": self.loaded,
            "simulated": False,
            "model_file": self._safe_model,
            "task": self.task,
            "classes": list(self.class_names.values()),
            "device": self.device_str,
            "cuda": self.device_str.startswith("cuda"),
            "half": self.half,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence,
            "require_transition": self.require_transition,
            "warmup_ms": round(self.warmup_ms, 1) if self.warmup_ms is not None else None,
            "last_inference_ms": self.last_inference_ms,
            "avg_inference_ms": round(self.avg_inference_ms, 2) if self.avg_inference_ms is not None else None,
            "inference_fps": self.inference_fps(),
            "inference_count": self.inference_count,
            "last_error": self._last_error,
        }
