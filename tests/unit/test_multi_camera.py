"""Tests for simultaneous multi-camera operation (hardware-independent).

Most logic is driven deterministically by calling ``worker._tick()`` directly
(no threads, no sleeps, ManualClock for fall timing). A few lifecycle tests
start/stop real worker threads with fast fakes to prove clean shutdown, a single
shared inference thread, and a bounded backlog.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

import pytest

from vytallink.alerts.base import AlertResult
from vytallink.common.clock import ManualClock, isoformat
from vytallink.common.types import Frame, RawDetection
from vytallink.config.cameras import CameraConfig
from vytallink.database.models import AlertRow
from vytallink.events.manager import EventManager
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState
from vytallink.vision.multi_camera import CameraWorker, MultiCameraMonitor, make_event_bridge

FALL_CLASSES = {"fall", "fallen", "lying"}
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _det(class_name: str, conf: float = 0.9) -> RawDetection:
    return RawDetection(
        timestamp=_T0, class_id=0, class_name=class_name, confidence=conf,
        bbox=(0.1, 0.1, 0.9, 0.9), source_id="cam", frame_id=1, metadata={},
    )


# --- fakes ----------------------------------------------------------------
class FakeCam:
    def __init__(self, *, source_id="camera_x", offline=False):
        self.source_id = source_id
        self._opened = False
        self._offline = offline
        self._fid = 0
        self.open_calls = 0
        self.close_calls = 0
        self._reconnects = 0

    def open(self):
        self.open_calls += 1
        self._opened = not self._offline

    def close(self):
        self.close_calls += 1
        self._opened = False

    def set_offline(self, v: bool):
        self._offline = v
        if v:
            self._opened = False

    def reconnect(self):
        self._offline = False
        self._opened = True
        self._reconnects += 1

    def read(self):
        if self._offline or not self._opened:
            return None
        self._fid += 1
        return Frame(frame_id=self._fid, timestamp=_T0, source_id=self.source_id,
                     width=2304, height=1296, image=object())

    def health(self):
        return {
            "opened": self._opened,
            "status": "ok" if self._opened else "down",
            "effective_fps": 0.0 if not self._opened else 15.0,
            "last_frame_age_seconds": None if not self._opened else 0.05,
            "reconnects": self._reconnects,
            "frames_dropped": 0,
            "resolution": [2304, 1296],
            "stale": not self._opened,
        }


class FakeDetector:
    name = "fake"

    def __init__(self, response=None):
        self.load_count = 0
        self.infer_count = 0
        self.infer_threads: set[str] = set()
        self.response = response if response is not None else []
        self.closed = False

    def load(self):
        self.load_count += 1
        time.sleep(0.001)

    def infer(self, frame):
        self.infer_threads.add(threading.current_thread().name)
        self.infer_count += 1
        return list(self.response)

    def close(self):
        self.closed = True


def _sm(clock, *, confirm=0.5, source="camera_1"):
    return FallEventStateMachine(
        confirm_seconds=confirm, clear_seconds=1.0, cooldown_seconds=0.0,
        source_device=source, clock=clock,
    )


def _worker(infer_fn, *, cam=None, clock=None, confirm=0.5, source="camera_1", smoother=None, max_fps=1000.0):
    clock = clock or ManualClock(start=_T0)
    cam = cam or FakeCam(source_id=source)
    sm = _sm(clock, confirm=confirm, source=source)
    cfg = CameraConfig(camera_id=source, host="192.168.42.71", username="u@e.com", password="pw")
    w = CameraWorker(cfg, cam, sm, sm.observe, infer_fn,
                     fall_class_set=FALL_CLASSES, confidence_threshold=0.5,
                     clock=clock, evidence_smoother=smoother, max_fps=max_fps)
    return w, cam, sm, clock


# --- deterministic _tick() logic ------------------------------------------
def test_healthy_worker_processes_and_counts_classes():
    w, cam, sm, clock = _worker(lambda f: [_det("sitting", 0.9)])
    cam.open()
    for _ in range(3):
        w._tick()
    h = w.health()
    assert h["connected"] is True
    assert h["frames_received"] == 3
    assert h["frames_processed"] == 3
    assert h["failed_reads"] == 0
    assert h["detected_classes"] == {"sitting": 3}
    assert h["fall_state"] == "normal"


def test_offline_camera_isolated_no_inference():
    w, cam, sm, clock = _worker(lambda f: [_det("fallen")])
    cam.open()           # open() while offline keeps it closed
    cam.set_offline(True)
    for _ in range(5):
        w._tick()
    h = w.health()
    assert h["connected"] is False
    assert h["failed_reads"] == 5
    assert h["frames_processed"] == 0      # never ran inference
    assert h["confirmed_falls"] == 0


def test_fall_confirms_on_one_camera():
    w, cam, sm, clock = _worker(lambda f: [_det("fallen", 0.95)], confirm=0.5)
    cam.open()
    w._tick()                  # NORMAL -> POSSIBLE
    assert sm.state is FallState.POSSIBLE_FALL
    clock.advance(0.6)         # exceed confirm window
    w._tick()                  # POSSIBLE -> CONFIRMED
    assert sm.state is FallState.CONFIRMED_FALL
    assert w.health()["confirmed_falls"] == 1


def test_no_duplicate_fall_events_for_sustained_fall():
    w, cam, sm, clock = _worker(lambda f: [_det("fallen", 0.95)], confirm=0.5)
    cam.open()
    w._tick()
    for _ in range(10):        # sustained fallen for many more frames
        clock.advance(0.6)
        w._tick()
    assert sm.state is FallState.CONFIRMED_FALL
    assert w.health()["confirmed_falls"] == 1       # exactly one event
    assert len(w.metrics(1.0)["events"]) == 1


def test_two_cameras_have_independent_fall_state():
    clock = ManualClock(start=_T0)
    w1, c1, sm1, _ = _worker(lambda f: [_det("fallen", 0.95)], cam=FakeCam(source_id="camera_1"),
                             clock=clock, source="camera_1", confirm=0.5)
    w2, c2, sm2, _ = _worker(lambda f: [_det("standing", 0.9)], cam=FakeCam(source_id="camera_2"),
                             clock=clock, source="camera_2", confirm=0.5)
    c1.open(); c2.open()
    for _ in range(4):
        w1._tick(); w2._tick(); clock.advance(0.6)
    assert sm1.state is FallState.CONFIRMED_FALL
    assert sm2.state is FallState.NORMAL          # camera 2 unaffected
    assert w2.health()["confirmed_falls"] == 0


def test_tick_error_is_isolated():
    def boom(frame):
        raise RuntimeError("inference exploded")
    w, cam, sm, clock = _worker(boom)
    cam.open()
    w._tick()  # must not raise
    assert w.health()["tick_errors"] == 1
    assert w.health()["frames_processed"] == 0


def test_health_has_no_credentials_or_url():
    w, cam, sm, clock = _worker(lambda f: [], source="camera_1")
    cam.open(); w._tick()
    blob = str(w.health())
    assert "u@e.com" not in blob and "pw" not in blob
    assert "rtsp://" not in blob and "192.168" not in blob
    for k in ("host", "url", "username", "password", "source"):
        assert k not in w.health()


def test_camera_config_safe_label_redacts():
    cfg = CameraConfig(camera_id="camera_1", host="192.168.42.71", username="u@e.com", password="Yeettheworld")
    assert "Yeettheworld" not in cfg.safe_label() and "u@e.com" not in cfg.safe_label()
    assert "%40" in cfg.rtsp_url()  # username @ encoded
    assert "Yeettheworld" in cfg.rtsp_url()  # creds only in the in-memory URL


# --- monitor lifecycle (real threads, fast fakes) -------------------------
def _monitor(detector, cams, *, max_fps=2000.0):
    clock = ManualClock(start=_T0)

    def factory(monitor):
        workers = []
        for i, cam in enumerate(cams, start=1):
            sm = _sm(clock, source=f"camera_{i}")
            cfg = CameraConfig(camera_id=f"camera_{i}", host="10.0.0.%d" % i, username="u@e.com", password="pw")
            workers.append(CameraWorker(cfg, cam, sm, sm.observe, monitor.infer,
                                        fall_class_set=FALL_CLASSES, confidence_threshold=0.5,
                                        clock=clock, max_fps=max_fps))
        return workers

    return MultiCameraMonitor(detector, factory, clock=clock)


def _wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_shared_model_loaded_once_and_single_inference_thread():
    det = FakeDetector(response=[_det("standing")])
    cams = [FakeCam(source_id="camera_1"), FakeCam(source_id="camera_2")]
    for c in cams:
        c.open()
    mon = _monitor(det, cams)
    mon.start()
    try:
        assert mon.model_load_count == 1
        assert _wait_until(lambda: det.infer_count > 20)
    finally:
        results = mon.stop()
    # ONE model load, and every inference ran on ONE dedicated thread.
    assert det.load_count == 1
    assert len(det.infer_threads) == 1
    assert all(results.values())                 # clean shutdown of both workers
    assert not mon.all_workers_alive()
    assert det.closed is True


def test_bounded_inference_backlog():
    det = FakeDetector(response=[])
    cams = [FakeCam(source_id="camera_1"), FakeCam(source_id="camera_2")]
    for c in cams:
        c.open()
    mon = _monitor(det, cams)
    mon.start()
    try:
        assert _wait_until(lambda: det.infer_count > 50)
    finally:
        mon.stop()
    # At most one in-flight frame per camera -> peak queue depth <= #cameras.
    assert mon.peak_queue_depth <= len(cams)


def test_one_camera_offline_other_stays_healthy():
    det = FakeDetector(response=[_det("standing")])
    good = FakeCam(source_id="camera_1")
    bad = FakeCam(source_id="camera_2", offline=True)
    good.open(); bad.open()
    mon = _monitor(det, [good, bad])
    mon.start()
    try:
        assert _wait_until(lambda: det.infer_count > 20)
        h = mon.health()   # snapshot WHILE running (stop() closes cameras)
    finally:
        mon.stop()
    assert h["cameras"]["camera_1"]["frames_processed"] > 0     # healthy one works
    assert h["cameras"]["camera_1"]["connected"] is True
    assert h["cameras"]["camera_2"]["connected"] is False       # offline one isolated
    assert h["cameras"]["camera_2"]["failed_reads"] > 0


def test_both_cameras_offline_no_crash_model_loads_once():
    det = FakeDetector()
    cams = [FakeCam(source_id="camera_1", offline=True), FakeCam(source_id="camera_2", offline=True)]
    mon = _monitor(det, cams)
    mon.start()
    try:
        assert _wait_until(lambda: all(w._failed_reads > 5 for w in mon.workers))
    finally:
        results = mon.stop()
    assert det.load_count == 1
    assert det.infer_count == 0
    assert all(results.values())


def test_reconnect_of_one_does_not_disturb_other():
    det = FakeDetector(response=[_det("standing")])
    flap = FakeCam(source_id="camera_1", offline=True)
    steady = FakeCam(source_id="camera_2")
    steady.open()
    mon = _monitor(det, [flap, steady])
    mon.start()
    try:
        assert _wait_until(lambda: mon.workers[1]._processed > 10)
        baseline = mon.workers[1]._processed
        flap.reconnect()                       # camera 1 comes online mid-run
        assert _wait_until(lambda: mon.workers[0]._processed > 5)
        assert _wait_until(lambda: mon.workers[1]._processed > baseline + 10)
    finally:
        results = mon.stop()
    assert mon.health()["cameras"]["camera_1"]["reconnects"] == 1
    assert all(results.values())               # both shut down cleanly (no deadlock)


class _FakeMonitor:
    """Stand-in for MultiCameraMonitor to exercise service health wiring."""

    def __init__(self, cam2_status="ok", cam2_connected=True):
        self._c2s = cam2_status
        self._c2c = cam2_connected

    def health(self):
        return {
            "mode": "rtsp_multi", "model_load_count": 1,
            "inference_queue_depth": 0, "inference_queue_peak": 2,
            "cameras": {
                "camera_1": {"connected": True, "status": "ok", "fps": 15.1,
                             "last_frame_age_ms": 60.0, "reconnects": 0, "fall_state": "normal"},
                "camera_2": {"connected": self._c2c, "status": self._c2s, "fps": 14.9,
                             "last_frame_age_ms": 70.0, "reconnects": 0, "fall_state": "normal"},
            },
        }


def _multi_service(monkeypatch, tmp_path):
    from vytallink.config import load_settings
    from vytallink.monitoring import MonitoringService

    monkeypatch.setenv("CAMERA_1_ENABLED", "true"); monkeypatch.setenv("CAMERA_1_HOST", "10.0.0.1")
    monkeypatch.setenv("CAMERA_2_ENABLED", "true"); monkeypatch.setenv("CAMERA_2_HOST", "10.0.0.2")
    settings = load_settings(
        env="development", vision_mode="simulation", detector_mode="simulation",
        database_path=str(tmp_path / "m.db"), log_dir=str(tmp_path / "l"),
        events_dir=str(tmp_path / "e"), clips_dir=str(tmp_path / "c"),
        disk_warning_percent=100.0,
    )
    return MonitoringService(settings)


def test_service_enters_multi_camera_mode(monkeypatch, tmp_path):
    svc = _multi_service(monkeypatch, tmp_path)
    assert svc.multi_camera_mode is True
    assert svc.simulation_mode is False           # multi-camera overrides simulation
    assert [c.camera_id for c in svc._camera_configs] == ["camera_1", "camera_2"]


def test_service_health_exposes_vision_block(monkeypatch, tmp_path):
    svc = _multi_service(monkeypatch, tmp_path)
    svc._multi_monitor = _FakeMonitor()
    svc._running = True
    h = svc.health()
    assert h["mode"] == "rtsp_multi"
    assert "vision" in h
    assert h["vision"]["mode"] == "rtsp_multi"
    assert set(h["vision"]["cameras"]) == {"camera_1", "camera_2"}
    assert h["camera"]["cameras_total"] == 2
    assert h["camera"]["cameras_connected"] == 2
    # No credentials/usernames/URLs anywhere in the health payload.
    blob = str(h)
    assert "rtsp://" not in blob and "@" not in blob


def test_service_health_degrades_when_one_camera_down(monkeypatch, tmp_path):
    svc = _multi_service(monkeypatch, tmp_path)
    svc._multi_monitor = _FakeMonitor(cam2_status="down", cam2_connected=False)
    svc._running = True
    svc.db.initialize()
    svc.detector.health = lambda: {"status": "ok", "name": "simulated", "loaded": True}  # loaded in real runs
    h = svc.health()
    assert h["camera"]["status"] == "degraded"     # one down -> degraded (other still up)
    assert h["overall"] == "degraded"


def test_build_helper_respects_enabled_count():
    from types import SimpleNamespace
    from vytallink.vision.multi_camera import build_multi_camera_monitor

    settings = SimpleNamespace(
        fall_confirm_seconds=2.0, fall_clear_seconds=3.0, alert_cooldown_seconds=30.0,
        fall_reconfirm_cooldown_seconds=0.0, evidence_hold_seconds=1.0,
        fall_class_set=FALL_CLASSES, confidence_threshold=0.55, detect_max_fps=12.0,
    )
    cfgs = [CameraConfig(camera_id="camera_1", host="10.0.0.1"),
            CameraConfig(camera_id="camera_2", host="10.0.0.2")]
    mon = build_multi_camera_monitor(settings, cfgs, detector=FakeDetector())
    assert len(mon.workers) == 2
    assert [w.camera_id for w in mon.workers] == ["camera_1", "camera_2"]


# ==========================================================================
# Persistence + alert integration: each camera drives its OWN EventManager
# (source_device=camera_id) through the SAME repos + dispatcher the
# single-camera path uses. Events persist to the DB and alerts dispatch, with
# per-camera attribution, duplicate suppression, and failure isolation.
# ==========================================================================


class _RecordingDispatcher:
    """Async dispatcher stand-in: records each AlertEvent + writes the alert row."""

    def __init__(self, repos, clock):
        self.repos = repos
        self.clock = clock
        self.calls = []  # the AlertEvents it was asked to deliver

    async def dispatch(self, alert):
        self.calls.append(alert)
        self.repos.alerts.record(
            AlertRow(
                event_uid=alert.event_uid, provider="fake",
                attempt_time=isoformat(alert.timestamp), success=True,
                response_metadata={"source_device": alert.source_device},
            )
        )
        return [AlertResult(provider="fake", success=True, attempt_time=self.clock.now())]

    async def aclose(self):
        pass


class _FailingDispatcher:
    """Dispatcher whose delivery raises — must be isolated, never crash observe."""

    def __init__(self):
        self.calls = 0

    async def dispatch(self, alert):
        self.calls += 1
        raise RuntimeError("alert backend unreachable")

    async def aclose(self):
        pass


def _camera_manager(repos, clock, *, camera_id, dispatcher, confirm=0.5):
    """A real EventManager for one camera, tagged with source_device=camera_id."""
    sm = FallEventStateMachine(
        confirm_seconds=confirm, clear_seconds=1.0, cooldown_seconds=0.0,
        source_device=camera_id, clock=clock,
    )
    em = EventManager(repos, sm, dispatcher, clock=clock, simulated=False)
    return em, sm


async def _confirm(em, clock, conf=0.92):
    await em.observe(True, conf)                       # NORMAL -> POSSIBLE
    clock.advance(em.sm.confirm_seconds + 0.05)
    return await em.observe(True, conf)                # POSSIBLE -> CONFIRMED (+1 alert)


@pytest.mark.asyncio
async def test_multicam_event_persists_with_camera_id_and_alerts(repos, manual_clock):
    disp = _RecordingDispatcher(repos, manual_clock)
    em, _ = _camera_manager(repos, manual_clock, camera_id="camera_1", dispatcher=disp)
    await _confirm(em, manual_clock)
    events = repos.events.list()
    assert len(events) == 1
    assert events[0].source_device == "camera_1"               # camera_id on the event
    assert events[0].state == FallState.CONFIRMED_FALL.value
    assert len(disp.calls) == 1
    assert disp.calls[0].source_device == "camera_1"           # camera_id on the alert
    assert repos.alerts.count() == 1


@pytest.mark.asyncio
async def test_multicam_two_cameras_persist_independent_events(repos, manual_clock):
    disp = _RecordingDispatcher(repos, manual_clock)
    em1, _ = _camera_manager(repos, manual_clock, camera_id="camera_1", dispatcher=disp)
    em2, _ = _camera_manager(repos, manual_clock, camera_id="camera_2", dispatcher=disp)
    await _confirm(em1, manual_clock)
    await _confirm(em2, manual_clock)
    assert sorted(e.source_device for e in repos.events.list()) == ["camera_1", "camera_2"]
    assert {a.source_device for a in disp.calls} == {"camera_1", "camera_2"}
    assert repos.events.count() == 2
    assert repos.alerts.count() == 2


@pytest.mark.asyncio
async def test_multicam_sustained_fall_one_event_one_alert(repos, manual_clock):
    disp = _RecordingDispatcher(repos, manual_clock)
    em, _ = _camera_manager(repos, manual_clock, camera_id="camera_1", dispatcher=disp)
    await _confirm(em, manual_clock)
    for _ in range(8):                                  # fall stays confirmed many frames
        manual_clock.advance(0.2)
        await em.observe(True, 0.95)
    assert repos.events.count() == 1                    # exactly one event
    assert repos.alerts.count() == 1                    # exactly one alert (no duplicates)
    assert len(disp.calls) == 1


@pytest.mark.asyncio
async def test_multicam_alert_failure_isolated_event_still_persists(repos, manual_clock):
    disp = _FailingDispatcher()
    em, _ = _camera_manager(repos, manual_clock, camera_id="camera_1", dispatcher=disp)
    await _confirm(em, manual_clock)                    # must NOT raise though dispatch raises
    assert disp.calls == 1
    assert repos.events.count() == 1                    # event persisted despite alert failure
    assert repos.events.list()[0].source_device == "camera_1"


# --- the sync->async bridge (its own running loop in a background thread) ---
@pytest.fixture
def bg_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="test-bridge-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        loop.close()


def test_event_bridge_returns_transitions_and_isolates_failure(bg_loop):
    class GoodEM:
        def __init__(self):
            self.seen = []

        async def observe(self, evidence, confidence):
            self.seen.append((evidence, confidence))
            return ["transition"]                       # stand-in transition list

    class RaisingEM:
        async def observe(self, evidence, confidence):
            raise RuntimeError("DB write failed")

    good = GoodEM()
    good_bridge = make_event_bridge(good, bg_loop, camera_id="camera_2", timeout=2.0)
    assert good_bridge(True, 0.5) == ["transition"]     # healthy bridge returns transitions
    assert good.seen == [(True, 0.5)]

    bad_bridge = make_event_bridge(RaisingEM(), bg_loop, camera_id="camera_1", timeout=2.0)
    assert bad_bridge(True, 0.9) == []                  # persistence failure isolated -> []
    # The healthy camera's bridge keeps working after the other one failed.
    assert good_bridge(False, 0.0) == ["transition"]


def test_event_bridge_times_out_without_deadlock(bg_loop):
    class HangingEM:
        async def observe(self, evidence, confidence):
            await asyncio.Event().wait()                # never completes

    bridge = make_event_bridge(HangingEM(), bg_loop, camera_id="camera_1", timeout=0.15)
    # A wedged loop must not deadlock the worker — the bridge returns [] on timeout.
    assert bridge(True, 0.9) == []


def test_one_camera_offline_other_records_alert_via_bridge(repos, manual_clock, bg_loop):
    """6f: one camera failing while the other persists + alerts, end to end through
    the real worker -> bridge -> EventManager -> repos/dispatcher path."""
    disp = _RecordingDispatcher(repos, manual_clock)
    bad_cam = FakeCam(source_id="camera_1", offline=True); bad_cam.open()
    good_cam = FakeCam(source_id="camera_2"); good_cam.open()

    em1, sm1 = _camera_manager(repos, manual_clock, camera_id="camera_1", dispatcher=disp)
    em2, sm2 = _camera_manager(repos, manual_clock, camera_id="camera_2", dispatcher=disp)
    cfg1 = CameraConfig(camera_id="camera_1", host="10.0.0.1", username="u@e.com", password="pw")
    cfg2 = CameraConfig(camera_id="camera_2", host="10.0.0.2", username="u@e.com", password="pw")
    w1 = CameraWorker(cfg1, bad_cam, sm1, make_event_bridge(em1, bg_loop, camera_id="camera_1"),
                      lambda f: [_det("fallen", 0.95)], fall_class_set=FALL_CLASSES,
                      confidence_threshold=0.5, clock=manual_clock, max_fps=1000.0)
    w2 = CameraWorker(cfg2, good_cam, sm2, make_event_bridge(em2, bg_loop, camera_id="camera_2"),
                      lambda f: [_det("fallen", 0.95)], fall_class_set=FALL_CLASSES,
                      confidence_threshold=0.5, clock=manual_clock, max_fps=1000.0)

    w1._tick(); w2._tick()                  # cam2: NORMAL->POSSIBLE; cam1: failed read (offline)
    manual_clock.advance(0.6)
    w1._tick(); w2._tick()                  # cam2: POSSIBLE->CONFIRMED -> persist + alert

    events = repos.events.list()
    assert len(events) == 1
    assert events[0].source_device == "camera_2"        # only the healthy camera produced one
    assert disp.calls and disp.calls[0].source_device == "camera_2"
    assert w1.health()["failed_reads"] >= 2             # offline camera kept failing, isolated
    assert w1.health()["confirmed_falls"] == 0
    assert w2.health()["confirmed_falls"] == 1


def test_clean_shutdown_with_event_bridge_active(repos, manual_clock, bg_loop):
    """Real worker threads driving the live bridge every tick must still stop
    cleanly — the loop stays free to drain in-flight observe() coroutines."""
    disp = _RecordingDispatcher(repos, manual_clock)
    cams = [FakeCam(source_id="camera_1"), FakeCam(source_id="camera_2")]
    for c in cams:
        c.open()

    def factory(monitor):
        workers = []
        for i, cam in enumerate(cams, start=1):
            cid = f"camera_{i}"
            sm = FallEventStateMachine(
                confirm_seconds=0.5, clear_seconds=1.0, cooldown_seconds=0.0,
                source_device=cid, clock=manual_clock,
            )
            em = EventManager(repos, sm, disp, clock=manual_clock, simulated=False)
            cfg = CameraConfig(camera_id=cid, host="10.0.0.%d" % i)
            workers.append(
                CameraWorker(cfg, cam, sm, make_event_bridge(em, bg_loop, camera_id=cid),
                             lambda f: [_det("standing")], fall_class_set=FALL_CLASSES,
                             confidence_threshold=0.5, clock=manual_clock, max_fps=500.0)
            )
        return workers

    mon = MultiCameraMonitor(FakeDetector(response=[_det("standing")]), factory, clock=manual_clock)
    mon.start()
    try:
        assert _wait_until(lambda: all(w._processed > 5 for w in mon.workers))
    finally:
        results = mon.stop()
    assert all(results.values())                        # both stopped cleanly (no bridge deadlock)
    assert not mon.all_workers_alive()
