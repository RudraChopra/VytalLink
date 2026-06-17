"""Deterministic simulated camera.

A *real* working provider (not a mock): it produces synthetic frames with
monotonic frame ids and real UTC timestamps. It carries no pixel data
(``image=None``) — Phase 1 needs none, and producing none avoids any privacy
surface. A dropout can be injected to exercise reconnection/backoff in tests.
"""

from __future__ import annotations

from typing import Any

from vytallink.common.logging_setup import get_logger
from vytallink.vision.base import CameraProvider

log = get_logger("vision.camera.simulated")


class SimulatedCamera(CameraProvider):
    description = "simulated camera (no live video)"

    def __init__(self, source_id: str = "camera-1", *, width: int = 640, height: int = 480, **kw):
        super().__init__(source_id, **kw)
        self.width = width
        self.height = height
        self._fail_remaining = 0

    def inject_dropout(self, count: int) -> None:
        """Make the next ``count`` reads fail (simulate a stream dropout)."""
        self._fail_remaining = max(0, int(count))

    def _open_source(self) -> None:
        # Nothing to open; the simulated source is always available once "open".
        self._fail_remaining = 0

    def _read_frame(self) -> Any | None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            return None  # treated by base as a failed read
        return (self.width, self.height, None)

    def _close_source(self) -> None:
        return None

    # -- health ------------------------------------------------------------
    def is_stale(self) -> bool:
        """A simulated camera carries no live video and produces a synthetic
        frame only when ``read()`` is called. Being *open with zero frames so
        far* is therefore normal, not stale — the simulation pipeline drives
        detections directly and need not pump synthetic frames. Once frames
        have actually flowed, fall back to the base frame-age staleness check
        (a stalled read loop is still a real, reportable condition)."""
        if self._last_frame_mono is None:
            return False
        return super().is_stale()

    def health(self) -> dict[str, Any]:
        h = super().health()
        # Explicitly mark this as simulated, no live video. These fields make it
        # clear to callers that frame_count == 0 / no pixel data is expected and
        # must not be read as a degraded real camera.
        h["simulated"] = True
        h["live_video_available"] = False
        return h
