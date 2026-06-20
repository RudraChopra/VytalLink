"""End-to-end: YOLO detector path -> evidence -> EventManager (DB + alerts).

Proves the real detection→evidence→state-machine→persistence/alert pipeline
behaves correctly, driven through the actual YoloFallDetector (fake weights) and
``detections_to_evidence`` rather than raw booleans. No GPU or camera required.
"""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import Frame
from vytallink.events.manager import EventManager
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState
from vytallink.vision.detector_base import detections_to_evidence
from vytallink.vision.detector_simulated import Scenario, SimulatedFallDetector
from tests._fakes import make_yolo_detector
from tests.unit.test_event_manager import FailingDispatcher, build_manager

FALL_CLASSES = {"fall", "fallen", "lying", "fall_detected", "person_fall"}
_fid = iter(range(1, 10_000_000))


def _evidence(det, clock, script_frame):
    det._model.set_script([script_frame])
    det._model._idx = 0
    f = Frame(frame_id=next(_fid), timestamp=clock.now(), source_id="cam",
              width=640, height=480, image=object())
    dets = det.infer(f)
    return detections_to_evidence(dets, FALL_CLASSES, 0.55)


async def _observe(mgr, det, clock, script_frame):
    evidence, conf = _evidence(det, clock, script_frame)
    return await mgr.observe(evidence, conf)


FALLEN = [(0, 0.92)]
STANDING = [(2, 0.85)]


@pytest.mark.asyncio
async def test_brief_evidence_creates_no_event(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=False)
    await _observe(mgr, det, manual_clock, FALLEN)      # POSSIBLE
    manual_clock.advance(0.5)                            # < confirm (2.0s)
    await _observe(mgr, det, manual_clock, STANDING)     # evidence gone -> dismissed
    assert repos.events.count() == 0
    assert len(disp.calls) == 0


@pytest.mark.asyncio
async def test_sustained_evidence_one_event_one_alert(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=False)
    await _observe(mgr, det, manual_clock, FALLEN)       # POSSIBLE
    manual_clock.advance(2.05)                           # past confirm
    await _observe(mgr, det, manual_clock, FALLEN)       # CONFIRMED + alert
    assert repos.events.count() == 1
    assert repos.events.list()[0].state == FallState.CONFIRMED_FALL.value
    assert len(disp.calls) == 1
    assert repos.alerts.count() == 1


@pytest.mark.asyncio
async def test_continued_detections_no_duplicate_event_or_alert(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=False)
    await _observe(mgr, det, manual_clock, FALLEN)
    manual_clock.advance(2.05)
    await _observe(mgr, det, manual_clock, FALLEN)       # CONFIRMED
    for _ in range(10):
        manual_clock.advance(0.2)
        await _observe(mgr, det, manual_clock, FALLEN)   # still down
    assert repos.events.count() == 1
    assert len(disp.calls) == 1
    assert repos.alerts.count() == 1


@pytest.mark.asyncio
async def test_recovery_then_independent_fall_two_events(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock, cooldown_seconds=30.0)
    det = make_yolo_detector(manual_clock, require_transition=False)
    # First fall.
    await _observe(mgr, det, manual_clock, FALLEN)
    manual_clock.advance(2.05)
    await _observe(mgr, det, manual_clock, FALLEN)       # CONFIRMED
    # Recover.
    await _observe(mgr, det, manual_clock, STANDING)     # RECOVERING
    manual_clock.advance(3.1)
    await _observe(mgr, det, manual_clock, STANDING)     # RESOLVED
    await _observe(mgr, det, manual_clock, STANDING)     # NORMAL
    assert repos.events.list()[0].state == FallState.RESOLVED.value
    # Independent second fall after cooldown.
    manual_clock.advance(31.0)
    await _observe(mgr, det, manual_clock, FALLEN)
    manual_clock.advance(2.05)
    await _observe(mgr, det, manual_clock, FALLEN)       # CONFIRMED again
    assert repos.events.count() == 2
    assert len(disp.calls) == 2
    assert repos.alerts.count() == 2


@pytest.mark.asyncio
async def test_failed_alert_does_not_suppress_next_fall(repos, manual_clock):
    import itertools

    counter = itertools.count(1)
    sm = FallEventStateMachine(
        confirm_seconds=2.0, clear_seconds=3.0, cooldown_seconds=30.0,
        clock=manual_clock, uid_factory=lambda: f"evt-{next(counter)}",
    )
    disp = FailingDispatcher(repos, manual_clock)
    mgr = EventManager(repos, sm, disp, clock=manual_clock, simulated=False)
    det = make_yolo_detector(manual_clock, require_transition=False)

    await _observe(mgr, det, manual_clock, FALLEN)
    manual_clock.advance(2.05)
    await _observe(mgr, det, manual_clock, FALLEN)       # CONFIRMED, delivery FAILS
    assert repos.alerts.count(success=True) == 0

    await _observe(mgr, det, manual_clock, STANDING)     # RECOVERING
    manual_clock.advance(3.1)
    await _observe(mgr, det, manual_clock, STANDING)     # RESOLVED
    await _observe(mgr, det, manual_clock, STANDING)     # NORMAL

    # Second real fall within the cooldown window must still attempt to alert.
    await _observe(mgr, det, manual_clock, FALLEN)
    manual_clock.advance(2.05)
    await _observe(mgr, det, manual_clock, FALLEN)
    assert repos.events.count() == 2
    assert len(disp.calls) == 2


@pytest.mark.asyncio
async def test_transition_gate_already_lying_creates_no_event(repos, manual_clock):
    """With the gate on, a person already on the floor never confirms a fall."""
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(12):
        await _observe(mgr, det, manual_clock, FALLEN)   # fallen, but no prior upright
        manual_clock.advance(0.3)
    assert repos.events.count() == 0
    assert len(disp.calls) == 0


@pytest.mark.asyncio
async def test_transition_gate_real_fall_creates_event(repos, manual_clock):
    """With the gate on, a genuine upright->fallen transition confirms a fall."""
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(3):                                   # establish upright
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.2)
    for _ in range(20):                                  # fall + stay down past confirm
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    assert repos.events.count() == 1
    assert len(disp.calls) == 1


@pytest.mark.asyncio
async def test_sparse_detection_with_smoother_confirms(repos, manual_clock):
    """A real fall detected only intermittently still confirms thanks to the
    live evidence smoother bridging the gaps (the state machine is untouched)."""
    from vytallink.vision.evidence import FallEvidenceSmoother

    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=False)
    smoother = FallEvidenceSmoother(1.0, clock=manual_clock)

    # Alternate fallen / no-detection (a sparse stream) for > confirm window.
    EMPTY: list = []
    for i in range(10):
        script = FALLEN if i % 2 == 0 else EMPTY
        ev, conf = _evidence(det, manual_clock, script)
        had_det = script is FALLEN
        ev, conf = smoother.update(ev, conf, had_detection=had_det, had_upright=False)
        await mgr.observe(ev, conf)
        manual_clock.advance(0.3)  # 10 * 0.3 = 3.0s > confirm (2.0s)

    assert repos.events.count() == 1
    assert repos.events.list()[0].state == FallState.CONFIRMED_FALL.value
    assert len(disp.calls) == 1


def test_simulation_detector_still_produces_evidence():
    """Sim regression: the simulated detector path remains intact."""
    clock = ManualClock()
    det = SimulatedFallDetector()
    det.load()
    det.set_scenario(Scenario.FALL)
    f = Frame(frame_id=1, timestamp=clock.now(), source_id="cam")
    ev, conf = detections_to_evidence(det.infer(f), FALL_CLASSES, 0.55)
    assert ev is True and conf > 0.55
    det.set_scenario(Scenario.NORMAL)
    ev2, _ = detections_to_evidence(det.infer(f), FALL_CLASSES, 0.55)
    assert ev2 is False
