"""HTTP relay camera provider: MJPEG streaming with snapshot-polling fallback.

Consumes a remote HTTP camera relay (e.g. a Jetson exposing a Tapo stream over
``/api/camera/stream`` and ``/api/camera/snapshot.jpg``) and presents it through
the standard latest-frame :class:`CameraProvider` interface, so the rest of the
pipeline (detector → events → dashboard) is unchanged.

Design (mirrors the RTSP grabber):

* A daemon **grabber thread** continuously pulls JPEG frames off the network and
  decodes them, keeping only the **newest** frame. A slow consumer can never
  build a backlog — old frames are intentionally dropped (counted). Networking
  and JPEG decoding therefore happen entirely off the API event loop.
* Per-connection **generation** fencing: a reconnect bumps the generation so a
  superseded thread can never publish a torn-down stream's frame as fresh, and a
  thread always closes its OWN response when it exits.
* **MJPEG is preferred**; if no stream URL is set (or only a snapshot URL is),
  the provider polls the snapshot endpoint at the target FPS instead.
* Bounded **connect + read timeouts** on every request; the base class drives
  **bounded exponential-backoff reconnection** when reads fail.

Security (non-negotiable): the stream/snapshot URLs and the optional bearer
token are NEVER logged, returned, or surfaced. Only a redacted
``scheme://host:port`` (:pyattr:`safe_source`) is ever exposed. The token is
sent ONLY via the ``Authorization: Bearer`` header. ``requests`` exceptions
embed the full URL, so they are caught and reduced to their type name before
they can reach a log, health payload, or the API. No frame is ever written to
disk.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Iterator

from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.sanitize import redact_http_endpoint
from vytallink.vision.base import CameraProvider

log = get_logger("vision.camera.http")

#: JPEG start-of-image / end-of-image markers. Within a valid JPEG entropy
#: stream every 0xFF byte is byte-stuffed (0xFF00) except real markers, and the
#: only markers allowed mid-scan are restart markers (0xFFD0–0xFFD7), so a bare
#: 0xFFD8/0xFFD9 reliably delimits a complete frame in an MJPEG byte stream.
_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"


def iter_jpeg_from_chunks(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Yield complete JPEG byte blobs from an arbitrary stream of byte chunks.

    Boundary-agnostic: scans for SOI…EOI, so it works regardless of the exact
    multipart boundary or part headers a relay emits. Pure and side-effect free
    so it is fully unit-testable without a network or a thread.
    """
    buf = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            start = buf.find(_SOI)
            if start == -1:
                # No frame start yet — drop everything but a trailing byte in
                # case it is the first half of an SOI split across chunks.
                if len(buf) > 1:
                    del buf[:-1]
                break
            end = buf.find(_EOI, start + 2)
            if end == -1:
                if start > 0:
                    del buf[:start]  # keep from SOI onward; wait for more bytes
                break
            yield bytes(buf[start : end + 2])
            del buf[: end + 2]


class HttpCamera(CameraProvider):
    """Latest-frame camera over an HTTP relay (MJPEG preferred, snapshot fallback)."""

    description = "HTTP relay camera"

    def __init__(
        self,
        *,
        stream_url: str = "",
        snapshot_url: str = "",
        bearer_token: str = "",
        source_id: str = "camera-http",
        prefer_mjpeg: bool = True,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        chunk_size: int = 8192,
        **kw: Any,
    ) -> None:
        super().__init__(source_id, **kw)
        self._stream_url = (stream_url or "").strip()
        self._snapshot_url = (snapshot_url or "").strip()
        self._token = (bearer_token or "").strip()  # secret; header-only, never logged
        self._prefer_mjpeg = bool(prefer_mjpeg)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self._chunk_size = int(chunk_size)
        self._transport = "mjpeg" if self._use_mjpeg() else "snapshot"

        # grabber state (guarded by _grab_lock)
        self._grab_thread: threading.Thread | None = None
        self._grab_stop = threading.Event()
        self._grab_lock = threading.Lock()
        self._grab_generation = 0  # bumped each (re)connect; fences old threads
        self._latest: Any = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._frames_grabbed = 0
        self._frames_consumed = 0
        self._frames_failed = 0  # cumulative failed reads (network/decode)
        self._last_grab_mono: float | None = None
        self._grab_error: str | None = None  # sanitized (type name only)
        self._resolution: tuple[int, int] | None = None
        self._response: Any = None
        self._jpeg_iter: Iterator[bytes] | None = None

    # -- safe identifiers --------------------------------------------------
    @property
    def safe_source(self) -> str:
        """``scheme://host:port`` — never the full URL, path, query, or token."""
        return redact_http_endpoint(self._stream_url or self._snapshot_url)

    def _use_mjpeg(self) -> bool:
        if self._prefer_mjpeg and self._stream_url:
            return True
        # No snapshot fallback available -> still try the stream URL if present.
        return bool(self._stream_url and not self._snapshot_url)

    def _auth_headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    # -- network (overridable for tests) -----------------------------------
    @staticmethod
    def _requests() -> Any:
        try:
            import requests  # noqa: WPS433 (lazy, optional dep)

            return requests
        except ImportError as exc:  # pragma: no cover - requests present in env
            raise CameraError(f"'requests' not available for HTTP camera: {exc}") from exc

    def _open_mjpeg(self) -> None:
        """Open the streaming MJPEG response and build the JPEG iterator.

        On any failure raise a sanitized :class:`CameraError` (no URL/token) so
        the base class records a safe message and applies backoff.
        """
        requests = self._requests()
        try:
            resp = requests.get(
                self._stream_url,
                stream=True,
                headers=self._auth_headers(),
                timeout=(self.connect_timeout, self.read_timeout),
            )
            resp.raise_for_status()
        except Exception as exc:
            # `requests` exceptions embed the full URL — reduce to type name.
            raise CameraError(
                f"Could not open MJPEG stream {self.safe_source}: {type(exc).__name__}"
            ) from None
        self._response = resp
        self._jpeg_iter = iter_jpeg_from_chunks(resp.iter_content(chunk_size=self._chunk_size))

    def _fetch_snapshot(self) -> bytes | None:
        requests = self._requests()
        resp = requests.get(
            self._snapshot_url,
            headers=self._auth_headers(),
            timeout=(self.connect_timeout, self.read_timeout),
        )
        try:
            resp.raise_for_status()
            return resp.content
        finally:
            resp.close()

    def _next_jpeg(self) -> bytes | None:
        """Return the next JPEG blob, or ``None`` on a (sanitized) failure."""
        try:
            if self._transport == "mjpeg":
                if self._jpeg_iter is None:
                    return None
                return next(self._jpeg_iter)
            return self._fetch_snapshot()
        except StopIteration:
            self._set_grab_error("stream_ended")
            return None
        except Exception as exc:  # never let a URL-bearing message escape
            self._set_grab_error(type(exc).__name__)
            return None

    # -- decode ------------------------------------------------------------
    @staticmethod
    def _decode(jpeg: bytes) -> Any | None:
        try:
            import cv2  # noqa: WPS433
            import numpy as np  # noqa: WPS433

            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img if img is not None else None
        except Exception:  # pragma: no cover - defensive
            return None

    # -- lifecycle hooks ---------------------------------------------------
    def _open_source(self) -> None:
        if not self._stream_url and not self._snapshot_url:
            raise CameraError(
                "HTTP camera requires CAMERA_HTTP_STREAM_URL and/or CAMERA_HTTP_SNAPSHOT_URL"
            )
        self._transport = "mjpeg" if self._use_mjpeg() else "snapshot"
        if self._transport == "mjpeg":
            log.info("Opening HTTP MJPEG source %s", self.safe_source)
            self._open_mjpeg()  # establishes the real connection (or raises safely)
        else:
            log.info("Opening HTTP snapshot source %s (polling)", self.safe_source)
            self._jpeg_iter = None  # snapshot mode fetches per grab

        # New connection: bump generation, reset per-connection grab state.
        with self._grab_lock:
            self._grab_generation += 1
            self._latest = None
            self._latest_seq = 0
            self._consumed_seq = 0
            self._last_grab_mono = None
            self._grab_error = None
        self._grab_stop.clear()
        self._start_grabber()

    def _start_grabber(self) -> None:
        """Start the background grabber thread (overridable in tests)."""
        gen = self._grab_generation
        self._grab_thread = threading.Thread(
            target=self._grab_loop, args=(gen,), name=f"http-grab-{self.source_id}", daemon=True
        )
        self._grab_thread.start()

    def _grab_loop(self, gen: int) -> None:
        # Snapshot mode paces itself to the target FPS; MJPEG blocks naturally on
        # the next frame. The loop exits as soon as it is superseded/stopped or a
        # read fails — the base class then reconnects with backoff. It ALWAYS
        # closes its OWN response on exit.
        interval = (1.0 / self.target_fps) if self._transport == "snapshot" else 0.0
        try:
            while gen == self._grab_generation and not self._grab_stop.is_set():
                t0 = self.clock.monotonic()
                if not self._grab_once(gen=gen):
                    break
                if interval:
                    elapsed = self.clock.monotonic() - t0
                    if elapsed < interval:
                        self._grab_stop.wait(interval - elapsed)
        finally:
            self._close_response()

    def _set_grab_error(self, msg: str) -> None:
        with self._grab_lock:
            self._grab_error = msg

    def _grab_once(self, gen: int | None = None) -> bool:
        """Pull + decode one frame into the latest slot, fenced by ``gen``.

        Factored out (no thread) so latest-frame/decoding logic is unit-testable.
        """
        gen = self._grab_generation if gen is None else gen
        jpeg = self._next_jpeg()
        if jpeg is None:
            with self._grab_lock:
                if gen == self._grab_generation:
                    self._frames_failed += 1
            return False
        image = self._decode(jpeg)
        if image is None:
            with self._grab_lock:
                if gen == self._grab_generation:
                    self._frames_failed += 1
                    self._grab_error = "jpeg_decode_failed"
            return False
        with self._grab_lock:
            if gen != self._grab_generation:
                return False  # superseded by a reconnect — discard
            self._latest = image
            self._latest_seq += 1
            self._frames_grabbed += 1
            self._last_grab_mono = self.clock.monotonic()
            if self._resolution is None:
                try:
                    h, w = image.shape[:2]
                    self._resolution = (int(w), int(h))
                except Exception:  # pragma: no cover - defensive
                    pass
        return True

    def _read_frame(self) -> Any | None:
        with self._grab_lock:
            latest = self._latest
            seq = self._latest_seq
            last_grab = self._last_grab_mono
            if latest is None:
                return None
            # Stale: grabber produced no new frame within the stale window.
            if last_grab is not None and (self.clock.monotonic() - last_grab) > self.stale_timeout:
                return None
            if seq > self._consumed_seq:
                self._consumed_seq = seq
                self._frames_consumed += 1
        return latest

    def _close_response(self) -> None:
        resp = self._response
        self._response = None
        if resp is not None:
            try:
                resp.close()
            except Exception:  # pragma: no cover - defensive
                pass

    def _close_source(self) -> None:
        # Supersede the grabber (bump generation + signal stop), then best-effort
        # join and close the response. Never join our own thread.
        self._grab_stop.set()
        with self._grab_lock:
            self._grab_generation += 1
            self._latest = None
            self._last_grab_mono = None
        thread = self._grab_thread
        self._grab_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._close_response()
        self._jpeg_iter = None

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
        h.update(
            {
                "safe_source": self.safe_source,
                "transport": self._transport,
                "resolution": list(self._resolution) if self._resolution else None,
                "frames_grabbed": self._frames_grabbed,
                "frames_consumed": self._frames_consumed,
                "frames_dropped": self.frames_dropped,
                "failed_reads": self._frames_failed,
                # Sanitized (type name / reason only) — never a URL or token.
                "last_grab_error": self._grab_error,
                "grabber_alive": bool(thread and thread.is_alive()),
            }
        )
        return h
