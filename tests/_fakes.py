"""Lightweight fakes so the real-hardware code paths are testable without a
GPU, real weights, or a live RTSP camera."""

from __future__ import annotations

from typing import Any


# --- Fake Ultralytics model ------------------------------------------------
class _Box:
    def __init__(self, cls: int, conf: float, xyxy=(10.0, 10.0, 50.0, 90.0)):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [list(xyxy)]


class _Result:
    def __init__(self, names: dict[int, str], boxes: list[_Box]):
        self.names = names
        self.boxes = boxes  # iterable of _Box


class FakeYoloModel:
    """Stands in for ``ultralytics.YOLO``. Feed it a per-frame script of
    ``[(class_id, confidence), ...]`` detections via :meth:`set_script`."""

    DEFAULT_NAMES = {0: "fallen", 1: "sitting", 2: "standing"}

    def __init__(self, names: dict[int, str] | None = None):
        self.names = names or dict(self.DEFAULT_NAMES)
        self.model = type("M", (), {"names": self.names})()
        self.task = "detect"
        self._script: list[list[tuple[int, float]]] = []
        self._idx = 0
        self.predict_calls = 0

    def set_script(self, frames: list[list[tuple[int, float]]]) -> None:
        self._script = frames
        self._idx = 0

    def to(self, device: str) -> "FakeYoloModel":
        return self

    def predict(self, image: Any, **kw: Any) -> list[_Result]:
        self.predict_calls += 1
        if self._script:
            spec = self._script[min(self._idx, len(self._script) - 1)]
            self._idx += 1
        else:
            spec = []
        boxes = [_Box(c, cf) for (c, cf) in spec]
        return [_Result(self.names, boxes)]


def make_yolo_detector(clock, *, require_transition: bool, confidence: float = 0.55):
    """Construct a YoloFallDetector wired to a FakeYoloModel (no load())."""
    from vytallink.vision.detector_yolo import YoloFallDetector

    det = YoloFallDetector(
        "/fake/models/fall_detection.pt",
        image_size=416,
        confidence=confidence,
        require_transition=require_transition,
        clock=clock,
        warmup=False,
    )
    det._model = FakeYoloModel()
    det.class_names = dict(FakeYoloModel.DEFAULT_NAMES)
    det.task = "detect"
    det.device_str = "cpu"
    return det


# --- Fake OpenCV VideoCapture ---------------------------------------------
class FakeCapture:
    """Minimal cv2.VideoCapture stand-in returning a scripted frame sequence.

    A ``None`` entry in ``frames`` yields ``(False, None)`` (a read failure).
    After the list is exhausted it keeps returning failures.
    """

    def __init__(self, frames: list[Any], opened: bool = True):
        self._frames = frames
        self._i = 0
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._i < len(self._frames):
            f = self._frames[self._i]
            self._i += 1
            return (f is not None, f)
        return (False, None)

    def set(self, *a) -> bool:
        return True

    def release(self) -> None:
        self.released = True
