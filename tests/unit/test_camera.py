"""Tests for the camera provider base + simulated camera (deterministic)."""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import HealthStatus
from vytallink.vision.simulated import SimulatedCamera


def test_simulated_camera_reads_frames(manual_clock: ManualClock):
    cam = SimulatedCamera(clock=manual_clock)
    cam.open()
    assert cam.is_open
    f1 = cam.read()
    assert f1 is not None
    assert f1.frame_id == 1
    assert f1.source_id == "camera-1"
    assert f1.has_image is False  # no pixel data in simulation
    f2 = cam.read()
    assert f2.frame_id == 2
    assert cam.frame_count == 2
    cam.close()
    assert not cam.is_open


def test_effective_fps_estimation(manual_clock: ManualClock):
    cam = SimulatedCamera(clock=manual_clock)
    cam.open()
    for _ in range(10):
        cam.read()
        manual_clock.advance(0.1)  # 10 fps
    assert cam.effective_fps() == pytest.approx(10.0, abs=0.5)


def test_stale_detection(manual_clock: ManualClock):
    cam = SimulatedCamera(clock=manual_clock, stale_timeout=5.0)
    cam.open()
    cam.read()
    assert cam.is_stale() is False
    manual_clock.advance(6.0)
    assert cam.is_stale() is True
    assert cam.status() is HealthStatus.DEGRADED


def test_dropout_then_recovery(manual_clock: ManualClock):
    cam = SimulatedCamera(clock=manual_clock, max_consecutive_failures=10)
    cam.open()
    assert cam.read() is not None  # frame 1
    cam.inject_dropout(2)
    assert cam.read() is None  # failure 1
    assert cam.status() is HealthStatus.DEGRADED
    assert cam.read() is None  # failure 2
    frame = cam.read()  # recovers
    assert frame is not None
    assert cam.status() is HealthStatus.OK


def test_excessive_failures_go_down_then_reconnect(manual_clock: ManualClock):
    cam = SimulatedCamera(
        clock=manual_clock, max_consecutive_failures=3, backoff_base=0.5, max_backoff=4.0
    )
    cam.open()
    cam.read()
    cam.inject_dropout(5)
    cam.read()  # fail 1
    cam.read()  # fail 2
    cam.read()  # fail 3 -> hits max, source closed
    assert cam.is_open is False
    assert cam.status() is HealthStatus.DOWN
    # Within backoff window, read returns None without reopening.
    assert cam.read() is None
    # After backoff elapses, it reconnects and recovers.
    manual_clock.advance(10.0)
    frame = cam.read()
    assert frame is not None
    assert cam.is_open is True


def test_health_payload_is_credential_safe(manual_clock: ManualClock):
    cam = SimulatedCamera(clock=manual_clock)
    cam.open()
    cam.read()
    h = cam.health()
    assert h["status"] == "ok"
    assert h["frame_count"] == 1
    assert h["opened"] is True
    assert "password" not in str(h).lower()


def test_open_with_no_frames_is_healthy(manual_clock: ManualClock):
    """Regression: a simulated camera that is open but has produced zero frames
    (the simulation pipeline drives detections directly) must report healthy —
    not stale/degraded — because it has no live video to be stale about."""
    cam = SimulatedCamera(clock=manual_clock, stale_timeout=5.0)
    cam.open()
    # No read() yet: frame_count == 0, no frame timestamp available.
    manual_clock.advance(100.0)  # well past stale_timeout, still no frames
    assert cam.frame_count == 0
    assert cam.is_stale() is False
    assert cam.status() is HealthStatus.OK
    h = cam.health()
    assert h["status"] == "ok"
    assert h["stale"] is False
    assert h["frame_count"] == 0
    assert h["simulated"] is True
    assert h["live_video_available"] is False
