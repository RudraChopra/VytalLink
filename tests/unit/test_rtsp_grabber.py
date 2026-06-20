"""Tests for the RTSP latest-frame grabber (no real camera/cv2 needed)."""

from __future__ import annotations

import numpy as np
import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import HealthStatus
from vytallink.vision.rtsp import RTSPCamera
from tests._fakes import FakeCapture


def _frame(val: int):
    a = np.zeros((480, 640, 3), dtype="uint8")
    a[0, 0, 0] = val  # tag so we can identify which frame
    return a


class _TestRTSP(RTSPCamera):
    """RTSPCamera that yields a scripted frame list instead of opening cv2."""

    def __init__(self, frames, clock, **kw):
        super().__init__(
            "rtsp://user:s3cret@cam.local:554/stream",
            source_id="cam",
            grab=True,  # latest-frame semantics...
            clock=clock,
            stale_timeout=2.0,
            **kw,
        )
        self._frames = frames

    def _create_capture(self):
        return FakeCapture(list(self._frames))

    def _start_grabber(self):
        pass  # ...but drive _grab_once() manually for determinism (no thread)


def test_latest_frame_and_dropped_count():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1), _frame(2), _frame(3)], clock)
    cam.open()
    # Grab three frames; consume once -> should get the NEWEST (3), dropping 2.
    assert cam._grab_once() and cam._grab_once() and cam._grab_once()
    latest = cam._read_frame()
    assert latest is not None and int(latest[0, 0, 0]) == 3
    h = cam.health()
    assert h["frames_grabbed"] == 3
    assert h["frames_consumed"] == 1
    assert h["frames_dropped"] == 2
    assert h["resolution"] == [640, 480]
    cam.close()


def test_read_none_before_any_frame():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    cam.open()
    assert cam._read_frame() is None  # nothing grabbed yet


def test_stale_returns_none_for_reconnect():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    cam.open()
    cam._grab_once()
    assert cam._read_frame() is not None
    clock.advance(3.0)  # > stale_timeout (2.0) with no new grab
    assert cam._read_frame() is None
    cam.close()


def test_rtsp_opened_without_frames_is_degraded():
    """Unlike the simulated camera, a real RTSP camera that opens but never
    delivers a frame is a genuine problem and must report degraded."""
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    cam.open()
    assert cam.frame_count == 0
    assert cam.is_stale() is True
    assert cam.status() is HealthStatus.DEGRADED
    cam.close()


def test_rtsp_goes_degraded_when_stale():
    """A real RTSP camera that stops delivering frames (frame age exceeds the
    stale timeout) must report degraded so caregivers know the feed is down."""
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    cam.open()
    cam._grab_once()
    assert cam.read() is not None  # public read records the last-frame timestamp
    assert cam.status() is HealthStatus.OK
    clock.advance(3.0)  # > stale_timeout (2.0) with no new frames
    assert cam.is_stale() is True
    assert cam.status() is HealthStatus.DEGRADED
    cam.close()


def test_credentials_never_in_safe_source_or_health():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    assert "s3cret" not in cam.safe_source
    assert "user" not in cam.safe_source
    assert "cam.local:554" in cam.safe_source
    cam.open()
    cam._grab_once()
    h = cam.health()
    assert "s3cret" not in str(h)


def test_reconnect_counter_via_base():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1)], clock)
    cam.open()
    cam.close()
    cam.open()  # second successful open => one reconnect
    assert cam.health()["reconnects"] == 1
    cam.close()


def test_superseded_generation_grab_is_discarded():
    """A reconnect bumps the generation; an old-generation grab must be dropped
    so a torn-down stream's frame can never be published as fresh."""
    clock = ManualClock()
    cam = _TestRTSP([_frame(1), _frame(2)], clock)
    cam.open()                       # generation -> 1
    assert cam._grab_once()          # gen 1 -> latest = frame 1
    assert int(cam._latest[0, 0, 0]) == 1
    old_gen = cam._grab_generation
    cam._grab_generation += 1        # simulate a reconnect superseding the old thread
    assert cam._grab_once(cap=cam._cap, gen=old_gen) is False  # discarded
    assert int(cam._latest[0, 0, 0]) == 1  # latest unchanged (frame 2 dropped)
    cam.close()


def test_frames_dropped_not_double_counted_on_reread():
    clock = ManualClock()
    cam = _TestRTSP([_frame(1), _frame(2), _frame(3)], clock)
    cam.open()
    cam._grab_once(); cam._grab_once(); cam._grab_once()  # grabbed=3, latest=frame3
    cam._read_frame()  # new frame -> consumed=1
    cam._read_frame()  # no new frame -> consumed must NOT increment again
    h = cam.health()
    assert h["frames_grabbed"] == 3
    assert h["frames_consumed"] == 1
    assert h["frames_dropped"] == 2  # stays non-negative and correct
    cam.close()


def test_base_read_wraps_grabbed_frame():
    clock = ManualClock()
    cam = _TestRTSP([_frame(7)], clock)
    cam.open()
    cam._grab_once()
    frame = cam.read()  # base.read() wraps _read_frame() into a Frame
    assert frame is not None
    assert frame.width == 640 and frame.height == 480
    assert frame.image is not None
    cam.close()


# --- transient-empty tolerance (flaky-link reconnect resilience) -----------
def _grabloop_cam(**over):
    # Real (system) clock so the time-based grace fires; short grace so the loop
    # terminates fast. No real RTSP — _grab_loop is driven with a FakeCapture.
    return RTSPCamera("rtsp://user:s3cret@cam.local:554/stream1", source_id="cam",
                      grab_failure_grace_seconds=over.get("grace", 0.05),
                      grab_max_transient_failures=over.get("max_fail", 100))


def test_grabber_rides_through_a_transient_empty_frame():
    cam = _grabloop_cam()
    cap = FakeCapture([_frame(1), _frame(2), None, _frame(3), _frame(4)])  # one transient empty
    cam._cap = cap
    cam._grab_generation = 1
    cam._grab_loop(cap, 1)
    # Did NOT break on the middle empty: all four real frames were grabbed.
    assert cam._frames_grabbed == 4
    assert cap.released


def test_grabber_breaks_on_sustained_failure():
    cam = _grabloop_cam(grace=0.05)
    cap = FakeCapture([])  # stream dead — always (False, None)
    cam._cap = cap
    cam._grab_generation = 1
    cam._grab_loop(cap, 1)   # must terminate via the grace bound, not hang
    assert cam._frames_grabbed == 0
    assert cap.released


def test_grabber_count_bound_terminates_even_without_clock_advance():
    cam = _grabloop_cam(grace=9999.0, max_fail=3)  # huge grace -> count bound must fire
    cap = FakeCapture([])
    cam._cap = cap
    cam._grab_generation = 1
    cam._grab_loop(cap, 1)
    assert cam._frames_grabbed == 0
    assert cap.released
