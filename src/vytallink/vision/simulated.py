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
