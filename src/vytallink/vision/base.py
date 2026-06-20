"""Camera provider interface with shared health, fps, and reconnection logic.

Concrete providers implement three hooks — ``_open_source``, ``_read_frame``,
``_close_source`` — and inherit:

* frame counting and effective-fps estimation,
* stale-stream detection,
* bounded-exponential-backoff reconnection on read failure,
* credential-safe logging (callers pass already-sanitized identifiers),
* clean shutdown.

``read()`` never raises; it returns a :class:`Frame` on success or ``None``
when no frame is available (the monitoring loop then skips detection and the
health status reflects the dropout).
"""

from __future__ import annotations

import abc
from collections import deque
from typing import Any

from vytallink.common.clock import Clock, SystemClock, isoformat
from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus

log = get_logger("vision.camera")


class CameraProvider(abc.ABC):
    def __init__(
        self,
        source_id: str,
        *,
        clock: Clock | None = None,
        target_fps: float = 10.0,
        stale_timeout: float = 5.0,
        backoff_base: float = 0.5,
        max_backoff: float = 30.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        self.source_id = source_id
        self.clock: Clock = clock or SystemClock()
        self.target_fps = max(0.1, float(target_fps))
        self.stale_timeout = float(stale_timeout)
        self.backoff_base = float(backoff_base)
        self.max_backoff = float(max_backoff)
        self.max_consecutive_failures = int(max_consecutive_failures)

        self._opened = False
        self._frame_count = 0
        self._last_frame_mono: float | None = None
        self._last_frame_time = None
        self._recent: deque[float] = deque(maxlen=30)
        self._consecutive_failures = 0
        self._next_retry_mono: float = 0.0
        self._last_error: str | None = None
        self._open_count = 0  # successful opens; reconnects = open_count - 1

    # -- describe ----------------------------------------------------------
    #: A human-readable, credential-safe description of the source.
    description: str = "camera"
    #: True when the provider maintains a background latest-frame buffer (so
    #: :meth:`peek_latest` is the source of truth and returning ``None`` means
    #: "no frame yet", not "this source has no buffer"). Sequential sources
    #: (e.g. a video file) leave this False and are read via :meth:`read`.
    has_latest_buffer: bool = False

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        try:
            self._open_source()
            self._opened = True
            self._open_count += 1
            self._consecutive_failures = 0
            self._last_error = None
            log.info("Camera %s opened (%s)", self.source_id, self.description)
        except Exception as exc:
            self._opened = False
            self._last_error = str(exc)
            log.error("Camera %s failed to open: %s", self.source_id, exc)
            raise CameraError(f"Failed to open camera {self.source_id}: {exc}") from exc

    def close(self) -> None:
        try:
            self._close_source()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Camera %s close error: %s", self.source_id, exc)
        finally:
            self._opened = False
            log.info("Camera %s closed", self.source_id)

    # -- read --------------------------------------------------------------
    def read(self) -> Frame | None:
        """Read one frame. Returns None when unavailable; never raises."""
        mono = self.clock.monotonic()

        if not self._opened:
            if mono < self._next_retry_mono:
                return None
            try:
                self.open()
            except CameraError:
                self._register_failure("reopen failed", mono)
                return None

        try:
            raw = self._read_frame()
        except Exception as exc:
            self._register_failure(str(exc), mono)
            return None

        if raw is None:
            self._register_failure("empty frame", mono)
            return None

        return self._register_success(raw, mono)

    def _register_success(self, raw: Any, mono: float) -> Frame:
        self._consecutive_failures = 0
        self._frame_count += 1
        now = self.clock.now()
        self._last_frame_mono = mono
        self._last_frame_time = now
        self._recent.append(mono)
        width, height, image = self._frame_dims(raw)
        return Frame(
            frame_id=self._frame_count,
            timestamp=now,
            source_id=self.source_id,
            width=width,
            height=height,
            image=image,
        )

    def _register_failure(self, error: str, mono: float) -> None:
        self._consecutive_failures += 1
        self._last_error = error
        backoff = min(self.backoff_base * (2 ** (self._consecutive_failures - 1)), self.max_backoff)
        self._next_retry_mono = mono + backoff
        if self._consecutive_failures >= self.max_consecutive_failures and self._opened:
            log.warning(
                "Camera %s: %d consecutive failures (%s); will reconnect with backoff %.1fs",
                self.source_id,
                self._consecutive_failures,
                error,
                backoff,
            )
            # Drop the source so the next read() attempts a clean reopen.
            self.close()

    # -- health ------------------------------------------------------------
    def effective_fps(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        if span <= 0:
            return 0.0
        return round((len(self._recent) - 1) / span, 2)

    def is_stale(self) -> bool:
        if self._last_frame_mono is None:
            return self._opened  # opened but never produced a frame
        return (self.clock.monotonic() - self._last_frame_mono) > self.stale_timeout

    def status(self) -> HealthStatus:
        if not self._opened and self._consecutive_failures >= self.max_consecutive_failures:
            return HealthStatus.DOWN
        if not self._opened:
            return HealthStatus.DOWN if self._last_error else HealthStatus.UNKNOWN
        if self._consecutive_failures > 0 or self.is_stale():
            return HealthStatus.DEGRADED
        return HealthStatus.OK

    def health(self) -> dict[str, Any]:
        last_age = None
        if self._last_frame_mono is not None:
            last_age = round(self.clock.monotonic() - self._last_frame_mono, 2)
        return {
            "status": self.status().value,
            "source_id": self.source_id,
            "description": self.description,
            "opened": self._opened,
            "frame_count": self._frame_count,
            "effective_fps": self.effective_fps(),
            "last_frame_time": isoformat(self._last_frame_time),
            "last_frame_age_seconds": last_age,
            "stale": self.is_stale(),
            "consecutive_failures": self._consecutive_failures,
            "reconnects": max(0, self._open_count - 1),
            "last_error": self._last_error,
        }

    # -- freshest-frame peek (non-consuming) -------------------------------
    def peek_latest(self) -> tuple[Any, int, float] | None:
        """Return ``(image, seq, age_seconds)`` for the freshest decoded frame
        WITHOUT consuming it or touching read counters, or ``None`` when no pixel
        frame is held (e.g. the simulated camera).

        ``seq`` increments once per genuinely new captured frame, so a consumer
        can de-duplicate (skip re-processing a frame the relay merely re-sent).
        ``age_seconds`` is measured from capture into the latest-frame buffer.
        Used by the dashboard/relay (serve the freshest frame, decoupled from the
        detection loop) and by the detection loop's stale-drop / de-dup logic.
        """
        return None

    # -- hooks for subclasses ----------------------------------------------
    @abc.abstractmethod
    def _open_source(self) -> None:  # pragma: no cover - interface
        ...

    @abc.abstractmethod
    def _read_frame(self) -> Any | None:  # pragma: no cover - interface
        ...

    @abc.abstractmethod
    def _close_source(self) -> None:  # pragma: no cover - interface
        ...

    def _frame_dims(self, raw: Any) -> tuple[int, int, Any]:
        """Return (width, height, image) for a raw read. Default: dimension-only
        sentinel tuple ``(w, h, None)`` from simulated sources. Real sources
        override to expose the numpy image array."""
        if isinstance(raw, tuple) and len(raw) == 3:
            return raw
        return (0, 0, None)
