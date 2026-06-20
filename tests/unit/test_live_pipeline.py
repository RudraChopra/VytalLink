"""Tests for the live-pipeline optimizations: freshest-frame peek, de-dup +
stale-drop detection loop, relay downscale, annotated preview (no double YOLO),
detection debug metrics, and the dashboard mode/device contract.

Deterministic and hardware-free: a fake latest-frame camera + a fake detector
drive the real MonitoringService logic.
"""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.device import device_label
from vytallink.common.types import Frame, HealthStatus, RawDetection
from vytallink.config import load_settings
from vytallink.database import Database
from vytallink.monitoring import MonitoringService


def _img(w=64, h=48, val=100):
    return np.full((h, w, 3), val, dtype="uint8")


def _det(name: str, conf: float = 0.9):
    return RawDetection(
        timestamp=ManualClock().now(), class_id=0, class_name=name, confidence=conf,
        bbox=(10.0, 10.0, 50.0, 90.0), source_id="cam", frame_id=1,
    )


class FakeLiveCamera:
    description = "fake relay camera"
    has_latest_buffer = True

    def __init__(self):
        self._peek = None  # (image, seq, age)
        self._status = HealthStatus.OK

    def set_peek(self, image, seq, age):
        self._peek = (image, seq, age)

    def peek_latest(self):
        return self._peek

    def read(self):
        # Drives liveness/reconnection in the real providers; the fake's inference
        # path is driven entirely by peek_latest(), so this is a no-op.
        return None

    def status(self):
        return self._status

    def health(self):
        return {
            "status": self._status.value, "effective_fps": 5.0, "frames_grabbed": 30,
            "frames_consumed": 10, "frames_dropped": 20, "failed_reads": 0,
            "reconnects": 0, "last_frame_age_seconds": 0.1, "safe_source": "http://host:5050",
        }

    def open(self): pass
    def close(self): pass


class FakeDetector:
    name = "fake"
    last_infer_ok = True

    def __init__(self, dets):
        self._dets = dets
        self.calls = 0

    def set(self, dets):
        self._dets = dets

    def infer(self, frame):
        self.calls += 1
        return list(self._dets)

    def inference_fps(self):
        return 5.0

    def health(self):
        return {
            "status": "ok", "name": "fake", "device": "mps", "device_label": "Apple MPS",
            "inference_fps": 5.0, "avg_inference_ms": 20.0, "inference_count": self.calls,
        }

    def load(self): pass
    def close(self): pass


def build_live_service(tmp_path, dets=None, *, event_clock=None, **over):
    s = load_settings(
        vision_mode="http_mjpeg", detector_mode="simulation", wearable_mode="simulation",
        alerts_enabled=False, database_path=str(tmp_path / "live.db"), **over,
    )
    db = Database(s.database_path, clock=ManualClock())
    svc = MonitoringService(s, db=db, event_clock=event_clock)
    svc.db.initialize()
    svc.camera = FakeLiveCamera()
    svc.detector = FakeDetector(dets if dets is not None else [_det("standing", 0.8)])
    return svc


# --- de-dup + stale drop ----------------------------------------------------
def test_live_loop_dedups_and_drops_stale(tmp_path):
    svc = build_live_service(tmp_path, dets=[_det("standing", 0.8)], detect_max_frame_age_seconds=1.0)
    cam, det = svc.camera, svc.detector

    cam.set_peek(_img(), seq=1, age=0.0)
    assert svc._detect_once_live() is not None      # fresh new frame -> processed
    assert det.calls == 1

    cam.set_peek(_img(), seq=1, age=0.0)
    assert svc._detect_once_live() is None           # same seq -> de-duplicated
    assert det.calls == 1                            # no second inference

    cam.set_peek(_img(), seq=2, age=5.0)             # new but too old
    assert svc._detect_once_live() is None           # dropped before inference
    assert det.calls == 1
    assert svc._frames_dropped_stale == 1

    cam.set_peek(_img(), seq=3, age=0.1)
    assert svc._detect_once_live() is not None        # fresh again -> processed
    assert det.calls == 2
    assert svc._frames_processed == 2                 # stale drop is NOT a processed frame


def test_no_frame_returns_none(tmp_path):
    svc = build_live_service(tmp_path)
    svc.camera.set_peek(None, 0, 0.0)  # peek_latest -> None
    svc.camera._peek = None
    assert svc._detect_once_live() is None
    assert svc.detector.calls == 0


# --- debug metrics ----------------------------------------------------------
def test_debug_metrics_track_classes_and_fallen(tmp_path):
    svc = build_live_service(tmp_path, dets=[_det("fallen", 0.92)], detect_max_frame_age_seconds=1.0)
    svc.camera.set_peek(_img(), seq=1, age=0.0)
    svc._detect_once_live()
    svc.detector.set([_det("standing", 0.7)])
    svc.camera.set_peek(_img(), seq=2, age=0.0)
    svc._detect_once_live()

    m = svc.debug_metrics()
    assert m["frames_processed"] == 2
    assert m["frames_with_fallen"] == 1
    assert m["class_counts"]["fallen"] == 1
    assert m["class_counts"]["standing"] == 1
    assert m["last_detections"][0]["class"] == "standing"
    assert m["detector"]["device_label"] == "Apple MPS"
    assert m["camera"]["frames_dropped_stale"] == 0
    assert "fallen" in m["fall_classes"]
    # No images / paths / credentials leak into the debug payload.
    blob = str(m)
    assert ".db" not in blob and "/Users/" not in blob


# --- annotated preview (uses existing result; never re-runs YOLO) -----------
def test_annotated_image_uses_existing_detections_without_reinference(tmp_path):
    svc = build_live_service(tmp_path, dets=[_det("fallen", 0.92)], detect_max_frame_age_seconds=1.0)
    svc.camera.set_peek(_img(200, 120), seq=1, age=0.0)
    svc._detect_once_live()
    calls_before = svc.detector.calls

    annotated = svc._build_annotated_image()
    assert annotated is not None
    assert annotated.shape == (120, 200, 3)
    assert svc.detector.calls == calls_before          # YOLO NOT run again for drawing
    # Annotation actually drew something (differs from the clean source frame).
    assert not np.array_equal(annotated, _img(200, 120))


# --- relay downscale --------------------------------------------------------
def test_relay_downscale_shrinks_dashboard_copy_only(tmp_path):
    import cv2
    svc = build_live_service(tmp_path, relay_width=960, relay_height=540)
    big = _img(2304, 1296, 70)
    svc.camera.set_peek(big, seq=1, age=0.0)
    jpeg = svc.latest_frame_jpeg(annotated=False)
    assert jpeg is not None
    out = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape[1] <= 960 and out.shape[0] <= 540   # downscaled
    # The detection input (peek frame) is untouched — still full resolution.
    assert svc.camera.peek_latest()[0].shape == (1296, 2304, 3)


def test_relay_downscale_disabled_keeps_native(tmp_path):
    import cv2
    svc = build_live_service(tmp_path, relay_width=0, relay_height=0)
    svc.camera.set_peek(_img(1280, 720), seq=1, age=0.0)
    out = cv2.imdecode(np.frombuffer(svc.latest_frame_jpeg(annotated=False), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape[1] == 1280 and out.shape[0] == 720


# --- dashboard mode/device contract -----------------------------------------
def test_live_health_reports_single_live_mode_and_mps_label(tmp_path):
    svc = build_live_service(tmp_path)
    svc._running = True
    h = svc.health()
    assert h["simulation"]["active"] is False          # -> dashboard shows LIVE only
    assert h["mode"] == "http_mjpeg"
    assert h["detector"]["device_label"] == "Apple MPS"
    assert "frames_dropped_stale" in h["camera"]


def test_simulation_health_reports_simulation_active(tmp_path):
    s = load_settings(vision_mode="simulation", detector_mode="simulation",
                      wearable_mode="simulation", database_path=str(tmp_path / "sim.db"))
    db = Database(s.database_path, clock=ManualClock())
    svc = MonitoringService(s, db=db)
    svc.db.initialize()
    svc._running = True
    h = svc.health()
    assert h["simulation"]["active"] is True            # -> dashboard shows SIMULATION only


# --- device label -----------------------------------------------------------
def test_device_label_mapping():
    assert device_label("mps") == "Apple MPS"
    assert device_label("cuda:0") == "CUDA"
    assert device_label("cpu") == "CPU"
    assert device_label("") == "—"


# --- settings validation ----------------------------------------------------
def test_relay_settings_validation():
    s = load_settings(relay_width=640, relay_height=360, relay_jpeg_quality=80,
                      relay_max_fps=8, detect_max_fps=15)
    assert (s.relay_width, s.relay_height, s.relay_jpeg_quality) == (640, 360, 80)
    with pytest.raises(Exception):
        load_settings(relay_jpeg_quality=0)
    with pytest.raises(Exception):
        load_settings(detect_max_fps=0)
    with pytest.raises(Exception):
        load_settings(relay_width=-5)


# --- frames flow into the event pipeline (alerts disabled) ------------------
@pytest.mark.asyncio
async def test_inference_pinned_to_single_dedicated_thread(tmp_path):
    """Regression: MPS/Metal is not thread-safe across threads. Every inference
    must run on ONE dedicated off-loop thread (asyncio.to_thread's multi-worker
    pool intermittently aborts with a Metal command-buffer assertion)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    svc = build_live_service(tmp_path, dets=[_det("standing", 0.8)],
                             detect_max_frame_age_seconds=10.0)
    svc._infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vytallink-infer")
    seen: list[str] = []
    base_infer = svc.detector.infer

    def recording(frame):
        seen.append(threading.current_thread().name)
        return base_infer(frame)

    svc.detector.infer = recording
    main_thread = threading.current_thread().name
    for i in range(1, 6):
        svc.camera.set_peek(_img(), seq=i, age=0.0)
        await svc._detect_and_observe_once()
    svc._infer_executor.shutdown(wait=True)

    assert len(seen) == 5
    assert all(name == seen[0] for name in seen)   # always the same thread
    assert seen[0] != main_thread                  # never the event-loop thread
    assert "vytallink-infer" in seen[0]            # the dedicated executor thread


@pytest.mark.asyncio
async def test_fresh_frames_confirm_event_with_alerts_disabled(tmp_path):
    evclk = ManualClock()
    svc = build_live_service(
        tmp_path, dets=[_det("fallen", 0.95)], event_clock=evclk,
        fall_confirm_seconds=2.0, detect_max_frame_age_seconds=10.0,
    )
    svc.camera.set_peek(_img(), seq=1, age=0.0)
    await svc._detect_and_observe_once()                 # POSSIBLE
    evclk.advance(2.1)
    svc.camera.set_peek(_img(), seq=2, age=0.0)
    await svc._detect_and_observe_once()                 # CONFIRMED
    assert svc.repos.events.count() == 1
    assert svc.repos.alerts.count() == 0                 # alerts disabled -> none delivered
    assert any(t["to"] == "confirmed_fall" for t in svc._transitions)
