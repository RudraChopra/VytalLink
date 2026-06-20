"""RTSP camera provider with a latest-frame background grabber.

Reproduces the legacy v1 ``FrameGrabber`` design: a daemon thread continuously
drains the RTSP stream (``cv2.VideoCapture`` with ``BUFFERSIZE=1`` + FFmpeg TCP +
a bounded open timeout) and keeps only the **newest** frame. Readers therefore
always get the freshest frame and a slow consumer can never build a stale
backlog — old frames are intentionally dropped (counted in ``frames_dropped``).

Thread safety / reconnection (hardened): each grabber is bound to a per-connection
**generation** token and **owns its own capture object**. On reconnect the base
class calls :meth:`close` → a new generation supersedes the old thread; the old
thread can never write the new connection's frame state (generation-fenced under
the lock) and releases its *own* capture when it exits — so a capture is never
released out from under an in-flight ``read()`` (no use-after-free), and a frame
from a torn-down stream can never be re-issued as fresh.

Security: the connection string may embed credentials
(``rtsp://user:pass@host/...``); **only the sanitized form is ever logged**.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Any

from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.sanitize import sanitize_url
from vytallink.vision.base import CameraProvider

log = get_logger("vision.camera.rtsp")


class RTSPCamera(CameraProvider):
    description = "RTSP camera"
    has_latest_buffer = True

    def __init__(
        self,
        connection_string: str,
        source_id: str = "camera-rtsp",
        *,
        open_timeout_us: int = 5_000_000,
        grab: bool = True,
        **kw,
    ):
        super().__init__(source_id, **kw)
        self._conn = connection_string  # may contain credentials; never log raw
        self.open_timeout_us = open_timeout_us
        self._use_grabber = grab
        self._cap: Any = None

        # grabber state (guarded by _grab_lock)
        self._grab_thread: threading.Thread | None = None
        self._grab_stop = threading.Event()
        self._grab_lock = threading.Lock()
        self._grab_generation = 0  # bumped each (re)connect; fences old threads
        self._latest: Any = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._frames_grabbed = 0
        self._frames_consumed = 0  # distinct latest frames delivered to a reader
        self._grab_marks: deque[float] = deque(maxlen=30)  # grab times -> ingest FPS
        self._last_grab_mono: float | None = None
        self._grab_error: str | None = None
        self._resolution: tuple[int, int] | None = None

    # -- safe identifiers --------------------------------------------------
    @property
    def safe_source(self) -> str:
        return sanitize_url(self._conn)

    # -- capture creation (overridable for tests) --------------------------
    def _create_capture(self) -> Any:
        try:
            import cv2  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - cv2 present on Jetson
            raise CameraError(f"OpenCV (cv2) not available: {exc}") from exc
        # FFmpeg: TCP transport + bounded timeout so a bad URL fails fast.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{self.open_timeout_us}",
        )
        cap = cv2.VideoCapture(self._conn, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - property may be unsupported
            pass
        return cap

    # -- lifecycle hooks ---------------------------------------------------
    def _open_source(self) -> None:
        if not self._conn:
            raise CameraError("RTSP connection string is empty")
        log.info("Opening RTSP source %s", self.safe_source)
        cap = self._create_capture()
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise CameraError(f"Could not open RTSP source {self.safe_source}")
        # New connection: bump the generation, reset per-connection grab state,
        # then publish the cap and start the grabber bound to this generation.
        with self._grab_lock:
            self._grab_generation += 1
            self._latest = None
            self._latest_seq = 0
            self._consumed_seq = 0
            self._last_grab_mono = None
            self._grab_error = None
        self._grab_stop.clear()
        self._cap = cap
        self._start_grabber()

    def _start_grabber(self) -> None:
        """Start the background grabber thread (overridable in tests)."""
        if not self._use_grabber:
            return
        cap = self._cap
        gen = self._grab_generation
        self._grab_thread = threading.Thread(
            target=self._grab_loop, args=(cap, gen), name=f"rtsp-grab-{self.source_id}", daemon=True
        )
        self._grab_thread.start()

    def _grab_once(self, cap: Any | None = None, gen: int | None = None) -> bool:
        """Grab one frame into the latest-frame slot, fenced by ``gen``.

        Writes to the shared latest-frame state ONLY while ``gen`` is still the
        current generation, so a superseded (reconnecting) thread can never
        corrupt the new connection's state. Factored out so the latest-frame
        logic is unit-testable without a thread.
        """
        cap = self._cap if cap is None else cap
        gen = self._grab_generation if gen is None else gen
        if cap is None:
            return False
        try:
            ok, frame = cap.read()
        except Exception as exc:
            with self._grab_lock:
                if gen == self._grab_generation:
                    self._grab_error = str(exc)
            return False
        if not ok or frame is None:
            with self._grab_lock:
                if gen == self._grab_generation:
                    self._grab_error = "empty frame"
            return False
        with self._grab_lock:
            if gen != self._grab_generation:
                return False  # superseded by a reconnect — discard this frame
            self._latest = frame
            self._latest_seq += 1
            self._frames_grabbed += 1
            self._last_grab_mono = self.clock.monotonic()
            self._grab_marks.append(self._last_grab_mono)
            if self._resolution is None:
                try:
                    h, w = frame.shape[:2]
                    self._resolution = (int(w), int(h))
                except Exception:  # pragma: no cover - defensive
                    pass
        return True

    def _grab_loop(self, cap: Any, gen: int) -> None:
        # Bound to (cap, gen): exits as soon as it is superseded or stopped, and
        # ALWAYS releases its OWN capture — never a capture owned by a newer
        # connection. This is what makes reconnects free of use-after-free.
        try:
            while gen == self._grab_generation and not self._grab_stop.is_set():
                if not self._grab_once(cap, gen):
                    break
        finally:
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                pass

    def peek_latest(self) -> Any | None:
        with self._grab_lock:
            if self._latest is None:
                return None
            grab = self._last_grab_mono
            age = 0.0 if grab is None else max(0.0, self.clock.monotonic() - grab)
            return (self._latest, self._latest_seq, age)

    # -- grabber-based liveness (independent of consumer read cadence) ------
    def effective_fps(self) -> float:
        with self._grab_lock:
            marks = list(self._grab_marks)
        if len(marks) < 2:
            return 0.0
        span = marks[-1] - marks[0]
        return round((len(marks) - 1) / span, 2) if span > 0 else 0.0

    def is_stale(self) -> bool:
        grab = self._last_grab_mono
        if grab is None:
            return self._opened
        return (self.clock.monotonic() - grab) > self.stale_timeout

    def _read_frame(self) -> Any | None:
        if not self._use_grabber:
            return self._grab_once_direct()

        with self._grab_lock:
            latest = self._latest
            seq = self._latest_seq
            last_grab = self._last_grab_mono
            if latest is None:
                return None
            # Stale: grabber produced no new frame within the stale window.
            if last_grab is not None and (self.clock.monotonic() - last_grab) > self.stale_timeout:
                return None
            # Count a consumption only when a genuinely NEW frame is delivered, so
            # frames_dropped = grabbed - distinct_consumed stays correct even when
            # a reader polls faster than the grabber (it just re-reads the latest).
            if seq > self._consumed_seq:
                self._consumed_seq = seq
                self._frames_consumed += 1
        return latest

    def _grab_once_direct(self) -> Any | None:
        cap = self._cap
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        with self._grab_lock:
            self._frames_grabbed += 1
            self._frames_consumed += 1
            self._last_grab_mono = self.clock.monotonic()
            if self._resolution is None:
                try:
                    h, w = frame.shape[:2]
                    self._resolution = (int(w), int(h))
                except Exception:  # pragma: no cover
                    pass
        return frame

    def _close_source(self) -> None:
        # Supersede the current grabber: bump the generation (so any in-flight
        # grab is discarded) and signal stop. We do NOT release the capture here —
        # the grabber thread owns and releases its own capture when it exits, so a
        # capture is never released while a read() is in flight.
        self._grab_stop.set()
        with self._grab_lock:
            self._grab_generation += 1
            self._latest = None
            self._last_grab_mono = None
        thread = self._grab_thread
        cap = self._cap
        self._cap = None
        if self._use_grabber:
            if thread is not None:
                # Best-effort brief join for the clean case; the thread self-releases
                # its capture regardless, and generation-fencing makes it harmless.
                thread.join(timeout=1.0)
                if not thread.is_alive():
                    self._grab_thread = None
        elif cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                pass

    def _frame_dims(self, raw: Any) -> tuple[int, int, Any]:
        try:
            h, w = raw.shape[:2]
            return int(w), int(h), raw
        except Exception:  # pragma: no cover - defensive
            return (0, 0, raw)

    # -- health ------------------------------------------------------------
    @property
    def frames_dropped(self) -> int:
        return max(0, self._frames_grabbed - self._frames_consumed)

    def health(self) -> dict[str, Any]:
        h = super().health()
        thread = self._grab_thread
        # Grab-based freshness (see HttpCamera.health) — independent of consumer
        # read cadence so it is not inflated during reconnects.
        grab = self._last_grab_mono
        h["last_frame_age_seconds"] = (
            None if grab is None else round(self.clock.monotonic() - grab, 2)
        )
        h.update(
            {
                "safe_source": self.safe_source,
                "resolution": list(self._resolution) if self._resolution else None,
                "frames_grabbed": self._frames_grabbed,
                "frames_consumed": self._frames_consumed,
                "frames_dropped": self.frames_dropped,
                # True only while the CURRENT grabber thread is running. A superseded
                # thread (still unwinding a blocked read) is not counted here, but it
                # is generation-fenced and self-releasing, so it cannot affect output.
                "grabber_alive": bool(thread and thread.is_alive()),
            }
        )
        return h
