"""Local video-file camera provider (uses OpenCV, imported lazily).

Useful for replaying recorded footage during development/testing. ``cv2`` is
imported inside the methods so the rest of the app (and the test suite) never
pays the import cost unless this provider is actually used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.vision.base import CameraProvider

log = get_logger("vision.camera.file")


class VideoFileCamera(CameraProvider):
    description = "local video file"

    def __init__(self, path: str, source_id: str = "camera-file", *, loop: bool = True, **kw):
        super().__init__(source_id, **kw)
        self.path = path
        self.loop = loop
        self._cap: Any = None

    def _open_source(self) -> None:
        if not Path(self.path).exists():
            raise CameraError(f"Video file not found: {self.path}")
        try:
            import cv2  # noqa: WPS433 (lazy import is intentional)
        except ImportError as exc:  # pragma: no cover - cv2 present on Jetson
            raise CameraError(f"OpenCV (cv2) not available: {exc}") from exc
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise CameraError(f"OpenCV could not open video file: {self.path}")
        self._cap = cap

    def _read_frame(self) -> Any | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop:
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    return frame
            return None
        return frame

    def _close_source(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _frame_dims(self, raw: Any) -> tuple[int, int, Any]:
        try:
            h, w = raw.shape[:2]
            return int(w), int(h), raw
        except Exception:  # pragma: no cover - defensive
            return (0, 0, raw)
