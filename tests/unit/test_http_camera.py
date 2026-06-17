"""Deterministic tests for the HTTP relay camera (no network, no real threads).

These drive the latest-frame / decode / staleness / reconnection logic via the
``_grab_once`` seam with a scripted JPEG source, exactly as the RTSP grabber
tests do — plus the pure MJPEG multipart parser, the bearer-header behavior, and
URL/token redaction. The Jetson IP/URL is NEVER hardcoded here; these tests use
example hostnames only.
"""

from __future__ import annotations

import numpy as np
import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import HealthStatus
from vytallink.config import VisionMode, load_settings
from vytallink.vision.http_source import HttpCamera, iter_jpeg_from_chunks

# Example endpoints only — host visible, full path must never be surfaced.
STREAM = "http://relay.example:5050/api/camera/stream"
SNAP = "http://relay.example:5050/api/camera/snapshot.jpg"
TOKEN = "s3cr3t-bearer-token"


def _jpeg(w: int = 64, h: int = 48, val: int = 120) -> bytes:
    import cv2

    img = np.full((h, w, 3), val, dtype="uint8")
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _ScriptedHttpCamera(HttpCamera):
    """HttpCamera that yields scripted JPEG bytes instead of doing network I/O.

    ``_open_mjpeg`` and the grabber thread are stubbed; ``_grab_once`` is driven
    manually for determinism (mirrors the RTSP grabber tests)."""

    def __init__(self, jpegs, clock, *, stream_url=STREAM, snapshot_url=SNAP,
                 token=TOKEN, fail_after=None, **kw):
        super().__init__(
            stream_url=stream_url, snapshot_url=snapshot_url, bearer_token=token,
            source_id="cam", clock=clock, stale_timeout=2.0, **kw,
        )
        self._jpegs = list(jpegs)
        self._ptr = 0
        self._fail_after = fail_after

    def _open_mjpeg(self) -> None:  # no real network
        self._jpeg_iter = iter(())

    def _start_grabber(self) -> None:  # drive _grab_once() by hand
        pass

    def _next_jpeg(self):
        if self._fail_after is not None and self._ptr >= self._fail_after:
            self._set_grab_error("ReadTimeout")
            return None
        if self._ptr < len(self._jpegs):
            j = self._jpegs[self._ptr]
            self._ptr += 1
            return j
        self._set_grab_error("stream_ended")
        return None


# --- pure MJPEG parser ------------------------------------------------------
def test_iter_jpeg_from_chunks_parses_multipart_across_chunk_boundaries():
    j1, j2 = _jpeg(val=10), _jpeg(val=200)
    body = (
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j1 + b"\r\n"
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j2 + b"\r\n"
    )
    # Feed in deliberately awkward 7-byte chunks to exercise the buffer.
    chunks = [body[i : i + 7] for i in range(0, len(body), 7)]
    out = list(iter_jpeg_from_chunks(chunks))
    assert out == [j1, j2]


def test_iter_jpeg_ignores_leading_noise_and_incomplete_tail():
    j1 = _jpeg(val=33)
    body = b"garbage-preamble" + b"--frame\r\n\r\n" + j1 + b"\r\n--frame\r\n\r\n\xff\xd8partial"
    out = list(iter_jpeg_from_chunks([body]))
    assert out == [j1]  # the partial trailing frame is not yielded


# --- decode + latest-frame semantics ---------------------------------------
def test_mjpeg_decodes_and_keeps_only_latest():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(val=10), _jpeg(val=20), _jpeg(val=30)], clock)
    cam.open()
    assert cam._grab_once() and cam._grab_once() and cam._grab_once()
    frame = cam.read()  # base wraps the newest grabbed frame
    assert frame is not None
    assert frame.width == 64 and frame.height == 48
    assert frame.image is not None  # real decoded ndarray
    h = cam.health()
    assert h["frames_grabbed"] == 3
    assert h["frames_consumed"] == 1
    assert h["frames_dropped"] == 2  # old frames intentionally discarded
    assert h["resolution"] == [64, 48]
    assert h["transport"] == "mjpeg"
    assert cam.status() is HealthStatus.OK
    cam.close()


def test_frames_dropped_not_double_counted_on_reread():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(val=1), _jpeg(val=2), _jpeg(val=3)], clock)
    cam.open()
    cam._grab_once(); cam._grab_once(); cam._grab_once()
    cam._read_frame()  # new frame -> consumed=1
    cam._read_frame()  # no new frame -> stays 1
    h = cam.health()
    assert h["frames_grabbed"] == 3 and h["frames_consumed"] == 1
    assert h["frames_dropped"] == 2
    cam.close()


# --- snapshot fallback ------------------------------------------------------
def test_snapshot_transport_selected_when_no_stream_url():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(val=77)], clock, stream_url="")
    assert cam._transport == "snapshot"
    cam.open()
    assert cam._grab_once()
    frame = cam.read()
    assert frame is not None and frame.image is not None
    assert cam.health()["transport"] == "snapshot"
    cam.close()


# --- bearer header ----------------------------------------------------------
def test_bearer_header_present_only_when_token_set():
    cam = HttpCamera(stream_url=STREAM, bearer_token=TOKEN)
    assert cam._auth_headers() == {"Authorization": f"Bearer {TOKEN}"}
    cam_no_token = HttpCamera(stream_url=STREAM)
    assert cam_no_token._auth_headers() == {}


# --- redaction --------------------------------------------------------------
def test_url_path_and_token_never_in_safe_source_or_health():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg()], clock)
    assert cam.safe_source == "http://relay.example:5050"
    assert TOKEN not in cam.safe_source
    assert "/api/camera/stream" not in cam.safe_source
    cam.open()
    cam._grab_once()
    blob = str(cam.health())
    assert TOKEN not in blob
    assert "/api/camera/stream" not in blob
    assert "/api/camera/snapshot.jpg" not in blob
    cam.close()


# --- staleness --------------------------------------------------------------
def test_stale_feed_degrades():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg()], clock)
    cam.open()
    cam._grab_once()
    assert cam.read() is not None
    assert cam.status() is HealthStatus.OK
    clock.advance(3.0)  # > stale_timeout (2.0) with no new frame
    assert cam.is_stale() is True
    assert cam._read_frame() is None  # stale frame is not re-served
    assert cam.status() is HealthStatus.DEGRADED
    cam.close()


def test_opened_without_frames_is_degraded():
    """A real relay camera that opens but never delivers a frame is degraded."""
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg()], clock)
    cam.open()
    assert cam.frame_count == 0
    assert cam.is_stale() is True
    assert cam.status() is HealthStatus.DEGRADED
    cam.close()


# --- failed reads + reconnect ----------------------------------------------
def test_failed_reads_tracked_and_reconnect_counted():
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(), _jpeg()], clock, fail_after=2,
                              max_consecutive_failures=3)
    cam.open()
    assert cam._grab_once() and cam._grab_once()  # two good frames
    assert cam._grab_once() is False              # simulated read timeout
    assert cam.health()["failed_reads"] == 1
    assert cam.health()["last_grab_error"] == "ReadTimeout"
    # A close()+open() cycle is one reconnect (tracked by the base class).
    cam.close()
    cam.open()
    assert cam.health()["reconnects"] == 1
    cam.close()


def test_superseded_generation_grab_is_discarded():
    """A reconnect bumps the generation; a stale-generation grab must be dropped."""
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(val=1), _jpeg(val=2)], clock)
    cam.open()
    assert cam._grab_once()                       # gen N -> frame 1 stored
    old_gen = cam._grab_generation
    cam._grab_generation += 1                     # simulate a superseding reconnect
    assert cam._grab_once(gen=old_gen) is False   # discarded
    assert cam._frames_grabbed == 1               # frame 2 was not published
    cam.close()


# --- settings-level redaction ----------------------------------------------
def test_settings_http_mode_redacts_url_and_token():
    s = load_settings(
        vision_mode="http_mjpeg",
        camera_http_stream_url=STREAM,
        camera_http_snapshot_url=SNAP,
        camera_http_bearer_token=TOKEN,
    )
    assert s.vision_mode == VisionMode.HTTP_MJPEG
    assert s.has_camera_target is True
    san = s.sanitized_camera_source()
    assert san == "http://relay.example:5050"
    assert "/api/camera/stream" not in san
    summary = s.safe_summary()
    blob = str(summary)
    assert TOKEN not in blob
    assert "/api/camera/stream" not in blob
    assert "/api/camera/snapshot.jpg" not in blob
    assert summary["camera_http_token"] == "***REDACTED***"
    assert summary["camera_source"] == san


def test_settings_http_token_absent_is_empty_redaction():
    s = load_settings(vision_mode="http_mjpeg", camera_http_stream_url=STREAM)
    assert s.safe_summary()["camera_http_token"] == ""


# --- frame-age is grab-based (regression: soak showed inflated read-based age) --
def test_frame_age_is_grab_based_not_read_based():
    """Health frame age must reflect GRAB freshness, not consumer read cadence,
    so it is not inflated during reconnects while frames are still flowing."""
    clock = ManualClock()
    cam = _ScriptedHttpCamera([_jpeg(), _jpeg()], clock)
    cam.open()
    cam._grab_once()
    clock.advance(2.0)  # time passes; no read() call at all
    assert cam.health()["last_frame_age_seconds"] == pytest.approx(2.0, abs=0.01)
    cam._grab_once()    # a fresh grab resets the age
    assert cam.health()["last_frame_age_seconds"] == pytest.approx(0.0, abs=0.01)
    cam.close()


def test_http_reconnect_backoff_is_bounded_low_for_lan():
    """A relay reconnect is cheap on a LAN; backoff is capped low so a flapping
    link recovers in seconds (overridable)."""
    assert HttpCamera(stream_url=STREAM).max_backoff <= 5.0
    assert HttpCamera(stream_url=STREAM, max_backoff=2.0).max_backoff == 2.0
