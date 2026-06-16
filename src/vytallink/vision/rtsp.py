"""RTSP camera provider (uses OpenCV/FFmpeg, imported lazily).

The connection string may embed credentials (``rtsp://user:pass@host/...``).
**Only the sanitized form is ever logged** — see :func:`sanitize_url`. A short
open/read timeout is requested through FFmpeg options so a bad URL fails fast
instead of blocking; the base class then applies bounded-backoff reconnection.

This provider is implemented cleanly but stays dormant in Phase 1 until a real
``CAMERA_SOURCE`` is supplied. We never guess or probe credentials.
"""

from __future__ import annotations

import os
from typing import Any

from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.sanitize import sanitize_url
from vytallink.vision.base import CameraProvider

log = get_logger("vision.camera.rtsp")


class RTSPCamera(CameraProvider):
    description = "RTSP camera"

    def __init__(
        self,
        connection_string: str,
        source_id: str = "camera-rtsp",
        *,
        open_timeout_us: int = 5_000_000,
        **kw,
    ):
        super().__init__(source_id, **kw)
        self._conn = connection_string  # may contain credentials; never log raw
        self.open_timeout_us = open_timeout_us
        self._cap: Any = None

    @property
    def safe_source(self) -> str:
        return sanitize_url(self._conn)

    def _open_source(self) -> None:
        if not self._conn:
            raise CameraError("RTSP connection string is empty")
        try:
            import cv2  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise CameraError(f"OpenCV (cv2) not available: {exc}") from exc

        # Ask FFmpeg to use TCP and a bounded timeout so a bad URL fails fast.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{self.open_timeout_us}",
        )
        log.info("Opening RTSP source %s", self.safe_source)
        cap = cv2.VideoCapture(self._conn, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - property may be unsupported
            pass
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Could not open RTSP source {self.safe_source}")
        self._cap = cap

    def _read_frame(self) -> Any | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
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
