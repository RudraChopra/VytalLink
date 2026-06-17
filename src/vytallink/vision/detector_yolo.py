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
from vytallink.common.device import (
    CPU_DEVICE,
    MPS_DEVICE,
    device_label,
    select_device,
    synchronize_device,
)
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
        min_fallen_box_area_frac: float = 0.0,
        reject_edge_clipped_fallen: bool = False,
        edge_margin_frac: float = 0.02,
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
        # Conservative false-positive box gates (0 / False => disabled). A fallen
        # box that is too small or clipped at a non-floor frame edge is recorded
        # for visibility but does NOT count as fall evidence.
        self.min_fallen_box_area_frac = float(min_fallen_box_area_frac)
        self.reject_edge_clipped_fallen = bool(reject_edge_clipped_fallen)
        self.edge_margin_frac = float(edge_margin_frac)
        #: Cumulative count of fallen boxes downgraded by each reason (debug).
        self.rejection_counts: dict[str, int] = {}

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
        #: Set when an op was unsupported on the selected accelerator (e.g. MPS)
        #: and the detector fell back to CPU. Surfaced so we never *silently*
        #: claim the accelerator was used.
        self.mps_fallback_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> tuple[str, bool]:
        # Single source of truth (common.device): probes CUDA → MPS → CPU and
        # falls back safely if a backend is present but not actually usable.
        dev = select_device(self._device_pref)
        cuda = dev.startswith("cuda")
        # NOTE: fp16 is OFF by default. On this Jetson (cuDNN 8.6 + torch 2.3)
        # several YOLO11 conv plans raise CUDNN_STATUS_NOT_SUPPORTED in half
        # precision and fall back to a slow path (~700 ms/frame vs ~35 ms fp32).
        # Enable explicitly (DETECTOR_HALF) only if a future cuDNN supports it.
        # half is CUDA-only (MPS uses fp32 here).
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
            self._warmup_with_fallback()

        log.info(
            "YOLO model '%s' loaded: task=%s classes=%s device=%s half=%s warmup=%sms",
            self._safe_model,
            self.task,
            self.class_names,
            self.device_str,
            self.half,
            round(self.warmup_ms, 1) if self.warmup_ms is not None else "n/a",
        )

    def _warmup_with_fallback(self) -> None:
        """Warm the model up once. If the selected accelerator (e.g. MPS) hits an
        unsupported op, fall back to CPU and warm up there — never crash load."""
        try:
            self.warmup_ms = self._run_warmup()
        except Exception as exc:
            if self._maybe_fallback_to_cpu(exc):
                try:
                    self.warmup_ms = self._run_warmup()
                except Exception as exc2:  # pragma: no cover - defensive
                    log.warning("CPU warmup after fallback failed: %s", type(exc2).__name__)
            else:  # pragma: no cover - defensive
                log.warning("YOLO warmup failed (continuing): %s", type(exc).__name__)

    def _maybe_fallback_to_cpu(self, exc: Exception) -> bool:
        """If running on an accelerator (MPS) and an op is unsupported, move the
        model to CPU and record the exact error. Returns True if it fell back.

        We never *silently* claim the accelerator: ``device_str`` becomes ``cpu``
        and ``mps_fallback_reason`` carries the original error for diagnostics.
        """
        if self.device_str != MPS_DEVICE:
            return False
        self.mps_fallback_reason = f"{type(exc).__name__}: {exc}"
        log.warning(
            "MPS op unsupported (%s); falling back to CPU for inference",
            self.mps_fallback_reason,
        )
        try:
            self._model.to(CPU_DEVICE)
        except Exception:  # pragma: no cover - defensive
            pass
        self.device_str = CPU_DEVICE
        self.half = False
        return True

    def _run_warmup(self) -> float:
        import numpy as np  # noqa: WPS433

        blank = np.zeros((self.image_size, self.image_size, 3), dtype="uint8")
        t0 = time.perf_counter()
        self._predict(blank)
        self._device_sync()
        return (time.perf_counter() - t0) * 1000.0

    def _device_sync(self) -> None:
        """Synchronize the active accelerator (CUDA/MPS) before reading a timer."""
        try:
            import torch  # noqa: WPS433

            synchronize_device(torch, self.device_str)
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
            # Never write annotated frames / labels to runs/ (footage off-disk).
            save=False,
            save_txt=False,
            save_conf=False,
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
            self._device_sync()
        except Exception as exc:
            # An unsupported MPS op surfaces here on the first real frame: fall
            # back to CPU and retry once so a fall is never missed mid-stream.
            if self._maybe_fallback_to_cpu(exc):
                try:
                    t0 = time.perf_counter()
                    results = self._predict(frame.image)
                    self._device_sync()
                except Exception as exc2:
                    self._last_error = f"{type(exc2).__name__}: {exc2}"
                    self.last_infer_ok = False
                    log.warning("YOLO inference failed after CPU fallback: %s", self._last_error)
                    return []
            else:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self.last_infer_ok = False
                log.warning("YOLO inference failed: %s", self._last_error)
                return []
        self._record_latency((time.perf_counter() - t0) * 1000.0)
        self.last_infer_ok = True
        self._last_error = None

        raw_boxes = self._extract_boxes(results)
        # Annotate each box with normalized geometry and (for fallen boxes) any
        # conservative rejection reason; only UNREJECTED fallen boxes count toward
        # the fall confidence the gate/threshold see.
        annotated = self._annotate_boxes(raw_boxes, frame.width or 0, frame.height or 0)
        best_fallen = max(
            (a["conf"] for a in annotated if a["name"] == "fallen" and a["rejection"] is None),
            default=0.0,
        )
        best_upright = max(
            (a["conf"] for a in annotated if a["name"] in ("sitting", "standing")),
            default=0.0,
        )
        has_det = bool(annotated)

        # Gate decides whether a fallen posture is a genuine fall *event*.
        if self.require_transition:
            fall_is_event = self._gate.observe(
                fallen_conf=best_fallen,
                upright_conf=best_upright,
                has_detection=has_det,
            )
        else:
            fall_is_event = best_fallen >= self.confidence

        return self._build_detections(frame, annotated, fall_is_event)

    def _box_geometry(self, bbox: tuple[float, float, float, float], w: int, h: int) -> dict[str, Any]:
        """Normalized geometry for analysis: bbox_norm, area_frac, aspect,
        vertical_center, and which frame edges the box is clipped against."""
        x1, y1, x2, y2 = bbox
        if w <= 0 or h <= 0:
            return {"bbox_norm": None, "area_frac": None, "aspect": None,
                    "vertical_center": None, "edges": []}
        nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
        bw, bh = max(0.0, nx2 - nx1), max(0.0, ny2 - ny1)
        m = self.edge_margin_frac
        edges = []
        if nx1 <= m:
            edges.append("left")
        if nx2 >= 1.0 - m:
            edges.append("right")
        if ny1 <= m:
            edges.append("top")
        if ny2 >= 1.0 - m:
            edges.append("bottom")
        return {
            "bbox_norm": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
            "area_frac": round(bw * bh, 5),
            "aspect": round(bw / bh, 3) if bh > 0 else None,
            "vertical_center": round((ny1 + ny2) / 2, 4),
            "edges": edges,
        }

    def _fallen_rejection(self, geom: dict[str, Any]) -> str | None:
        """Conservative reason this fallen box should NOT count as fall evidence,
        or None. Bottom-edge clipping is allowed (real falls land low)."""
        area = geom.get("area_frac")
        if self.min_fallen_box_area_frac > 0 and area is not None and area < self.min_fallen_box_area_frac:
            return "too_small"
        if self.reject_edge_clipped_fallen:
            # Only non-floor edges indicate a partial person entering/leaving the
            # frame; a fallen person legitimately touches the bottom edge.
            bad = [e for e in geom.get("edges", []) if e in ("left", "right", "top")]
            if bad:
                return "edge_clipped_" + "".join(sorted(e[0] for e in bad))
        return None

    def _annotate_boxes(self, raw_boxes, w: int, h: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cls_id, name, conf, bbox in raw_boxes:
            geom = self._box_geometry(bbox, w, h)
            rejection = self._fallen_rejection(geom) if name == "fallen" else None
            if rejection is not None:
                self.rejection_counts[rejection] = self.rejection_counts.get(rejection, 0) + 1
            out.append({"cls_id": cls_id, "name": name, "conf": conf, "bbox": bbox,
                        "geom": geom, "rejection": rejection})
        return out

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

    def _build_detections(
        self,
        frame: Frame,
        annotated: list[dict[str, Any]],
        fall_is_event: bool,
    ) -> list[RawDetection]:
        detections: list[RawDetection] = []
        for a in annotated:
            name, cls_id, conf, bbox = a["name"], a["cls_id"], a["conf"], a["bbox"]
            rejection, geom = a["rejection"], a["geom"]
            class_name = name
            # A fallen box is surfaced as the non-evidence ``fallen_posture`` class
            # when a conservative box gate rejected it, OR (transition gate on) when
            # it is not part of a confirmed upright→fallen transition.
            if name == "fallen" and (
                rejection is not None or (self.require_transition and not fall_is_event)
            ):
                class_name = FALLEN_POSTURE_CLASS
            metadata: dict[str, Any] = {"simulated": False, "raw_class": name, **geom}
            if rejection is not None:
                metadata["rejection"] = rejection
            detections.append(
                RawDetection(
                    timestamp=frame.timestamp,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    source_id=frame.source_id,
                    frame_id=frame.frame_id,
                    metadata=metadata,
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
            "device_label": device_label(self.device_str),
            "cuda": self.device_str.startswith("cuda"),
            "mps": self.device_str == MPS_DEVICE,
            "mps_fallback": self.mps_fallback_reason,
            "half": self.half,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence,
            "require_transition": self.require_transition,
            "min_fallen_box_area_frac": self.min_fallen_box_area_frac,
            "reject_edge_clipped_fallen": self.reject_edge_clipped_fallen,
            "rejection_counts": dict(self.rejection_counts),
            "warmup_ms": round(self.warmup_ms, 1) if self.warmup_ms is not None else None,
            "last_inference_ms": self.last_inference_ms,
            "avg_inference_ms": round(self.avg_inference_ms, 2) if self.avg_inference_ms is not None else None,
            "inference_fps": self.inference_fps(),
            "inference_count": self.inference_count,
            "last_error": self._last_error,
        }
