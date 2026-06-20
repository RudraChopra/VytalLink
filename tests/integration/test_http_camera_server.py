"""Integration tests for the HTTP relay camera against a real local HTTP server.

A tiny ``ThreadingHTTPServer`` on 127.0.0.1 stands in for the Jetson relay and
serves both a multipart MJPEG stream and JPEG snapshots, optionally requiring a
bearer token. This exercises the real ``requests`` + multipart-parse + grabber-
thread path end to end without any hardware. (127.0.0.1 is the loopback test
server — the real Jetson IP is never hardcoded.)

It also covers the ALERTS_ENABLED master switch and proves an HTTP-decoded frame
flows into the real YOLO event pipeline.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from vytallink.common.clock import ManualClock, SystemClock
from vytallink.common.errors import CameraError
from vytallink.common.types import HealthStatus
from vytallink.config import load_settings
from vytallink.alerts.factory import build_dispatcher
from vytallink.vision.detector_base import detections_to_evidence
from vytallink.vision.http_source import HttpCamera

from tests.unit.test_http_camera import _ScriptedHttpCamera, _jpeg
from tests.unit.test_event_manager import build_manager

FALL_CLASSES = {"fall", "fallen", "lying", "fall_detected", "person_fall"}


def _wait(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _RelayHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence test server logging
        pass

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if not token:
            return True
        return self.headers.get("Authorization") == f"Bearer {token}"

    def do_GET(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path.startswith("/snapshot"):
            body = _jpeg(val=77)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for i in range(self.server.frames_per_stream):  # type: ignore[attr-defined]
                    j = _jpeg(val=(i * 9) % 240)
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(404)
        self.end_headers()


class _Relay:
    def __init__(self, *, token: str = "", frames_per_stream: int = 1000):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RelayHandler)
        self.httpd.token = token  # type: ignore[attr-defined]
        self.httpd.frames_per_stream = frames_per_stream  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def stream_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream"

    @property
    def snapshot_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/snapshot.jpg"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def relay():
    r = _Relay()
    yield r
    r.close()


# --- MJPEG end-to-end -------------------------------------------------------
def test_mjpeg_stream_delivers_real_frames(relay):
    cam = HttpCamera(stream_url=relay.stream_url, clock=SystemClock(), stale_timeout=2.0)
    cam.open()
    try:
        assert _wait(lambda: cam.frame_count > 0 or cam.read() is not None)
        frame = cam.read()
        assert frame is not None and frame.image is not None
        assert frame.width == 64 and frame.height == 48
        h = cam.health()
        assert h["transport"] == "mjpeg"
        assert h["resolution"] == [64, 48]
        assert h["frames_grabbed"] >= 1
        assert cam.status() in (HealthStatus.OK, HealthStatus.DEGRADED)
    finally:
        cam.close()


# --- snapshot fallback end-to-end ------------------------------------------
def test_snapshot_polling_delivers_real_frames(relay):
    cam = HttpCamera(
        snapshot_url=relay.snapshot_url, clock=SystemClock(),
        stale_timeout=2.0, target_fps=20.0,
    )
    assert cam._transport == "snapshot"
    cam.open()
    try:
        assert _wait(lambda: cam.read() is not None)
        frame = cam.read()
        assert frame is not None and frame.image is not None
        assert cam.health()["transport"] == "snapshot"
    finally:
        cam.close()


# --- bearer enforcement -----------------------------------------------------
def test_bearer_required_blocks_without_token_and_allows_with_token():
    secured = _Relay(token="hunter2")
    try:
        # Without the token the relay returns 401; the stream fails to open and
        # the error must be sanitized (no token, no full path/URL).
        bad = HttpCamera(stream_url=secured.stream_url, clock=SystemClock(), stale_timeout=1.0)
        with pytest.raises(CameraError) as ei:
            bad.open()
        msg = str(ei.value)
        assert "hunter2" not in msg
        assert "/stream" not in msg  # full path never leaked
        assert bad.frame_count == 0
        bad.close()

        # With the correct token frames flow.
        good = HttpCamera(
            stream_url=secured.stream_url, bearer_token="hunter2",
            clock=SystemClock(), stale_timeout=2.0,
        )
        good.open()
        try:
            assert _wait(lambda: good.read() is not None, timeout=4.0)
        finally:
            good.close()
    finally:
        secured.close()


# --- reconnect --------------------------------------------------------------
def test_stream_reconnects_when_connection_ends(relay):
    # Each GET serves a handful of frames then ends; the camera must reconnect.
    relay.httpd.frames_per_stream = 3  # type: ignore[attr-defined]
    cam = HttpCamera(
        stream_url=relay.stream_url, clock=SystemClock(),
        stale_timeout=0.2, max_consecutive_failures=2,
        backoff_base=0.05, max_backoff=0.2,
    )
    cam.open()
    try:
        # Drive reads (as the monitor loop would) and wait for >=1 reconnect.
        def _poll_once_and_check():
            cam.read()
            return cam.health()["reconnects"] >= 1

        assert _wait(_poll_once_and_check, timeout=8.0, interval=0.03)
        assert cam.health()["frames_grabbed"] >= 3
    finally:
        cam.close()


# --- ALERTS_ENABLED master switch ------------------------------------------
def test_alerts_disabled_builds_no_providers(repos):
    s = load_settings(alerts_enabled=False, console_alerts_enabled=True,
                      webhook_url="https://hooks.example.com/x")
    disp = build_dispatcher(s, repos)
    assert disp.providers == []
    assert disp.provider_names == []


def test_alerts_enabled_builds_console_provider(repos):
    s = load_settings(alerts_enabled=True, console_alerts_enabled=True)
    disp = build_dispatcher(s, repos)
    assert "console" in disp.provider_names


# --- HTTP frame -> YOLO event pipeline -------------------------------------
@pytest.mark.asyncio
async def test_http_decoded_frame_enters_yolo_event_pipeline(repos, manual_clock):
    """A frame decoded by the HTTP camera feeds the real detector→evidence→
    state-machine path and confirms a fall (FakeYoloModel scripts the posture)."""
    from tests._fakes import make_yolo_detector

    FALLEN = [(0, 0.92)]
    cam = _ScriptedHttpCamera([_jpeg(val=15)] * 6, manual_clock)
    cam.open()
    det = make_yolo_detector(manual_clock, require_transition=False)
    mgr, disp = build_manager(repos, manual_clock)

    def _http_fall_evidence():
        assert cam._grab_once()
        frame = cam.read()
        assert frame is not None and frame.image is not None  # decoded by HttpCamera
        det._model.set_script([FALLEN])
        det._model._idx = 0
        dets = det.infer(frame)
        return detections_to_evidence(dets, FALL_CLASSES, 0.55)

    ev, conf = _http_fall_evidence()
    await mgr.observe(ev, conf)              # POSSIBLE
    manual_clock.advance(2.05)               # past confirm window
    ev, conf = _http_fall_evidence()
    await mgr.observe(ev, conf)              # CONFIRMED

    assert repos.events.count() == 1
    assert len(disp.calls) == 1
    cam.close()
