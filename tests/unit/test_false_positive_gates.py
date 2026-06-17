"""False-positive reduction regression tests (Phase H).

Covers the transition gate, the conservative detector box gates (size / edge),
geometry metadata, and the state-machine reconfirm cooldown. Deterministic with
ManualClock + the fake YOLO model — no GPU, camera, or network.
"""

from __future__ import annotations

import itertools

import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import Frame
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState
from vytallink.vision.detector_base import detections_to_evidence
from tests._fakes import make_yolo_detector
from tests.unit.test_event_manager import build_manager
from tests.unit.test_live_pipeline import build_live_service, _img, _det

FALL = {"fall", "fallen", "lying", "fall_detected", "person_fall"}
_fid = itertools.count(1)
FALLEN = [(0, 0.92)]
SITTING = [(1, 0.85)]
STANDING = [(2, 0.85)]


def _evidence(det, clock, script):
    det._model.set_script([script])
    det._model._idx = 0
    f = Frame(frame_id=next(_fid), timestamp=clock.now(), source_id="cam",
              width=640, height=480, image=object())
    return detections_to_evidence(det.infer(f), FALL, 0.55)


async def _observe(mgr, det, clock, script):
    ev, conf = _evidence(det, clock, script)
    return await mgr.observe(ev, conf)


def _frame(det, clock, script):
    det._model.set_script([script])
    det._model._idx = 0
    f = Frame(frame_id=next(_fid), timestamp=clock.now(), source_id="cam",
              width=640, height=480, image=object())
    return det.infer(f)


# === gate behaviour (DETECTOR_REQUIRE_TRANSITION) ===========================
@pytest.mark.asyncio
async def test_standing_to_fallen_transition_confirms(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(3):
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.2)
    for _ in range(20):
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    assert repos.events.count() == 1


@pytest.mark.asyncio
async def test_sitting_continuously_does_not_confirm(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(30):
        await _observe(mgr, det, manual_clock, SITTING)
        manual_clock.advance(0.3)
    assert repos.events.count() == 0


@pytest.mark.asyncio
async def test_already_lying_at_startup_does_not_repeatedly_trigger(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(40):  # person already on the floor, no prior upright, for a long time
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    assert repos.events.count() == 0


@pytest.mark.asyncio
async def test_one_continuous_low_posture_at_most_one_event(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(3):
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.2)
    for _ in range(60):  # ~18s continuously down after one transition
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    assert repos.events.count() == 1


@pytest.mark.asyncio
async def test_recovery_allows_future_independent_event(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock, cooldown_seconds=0.0)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(3):
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.2)
    for _ in range(10):
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    for _ in range(20):  # genuine recovery to standing (resolves + re-arms gate)
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.3)
    for _ in range(10):
        await _observe(mgr, det, manual_clock, FALLEN)
        manual_clock.advance(0.3)
    assert repos.events.count() == 2


@pytest.mark.asyncio
async def test_low_confidence_flicker_does_not_confirm(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    det = make_yolo_detector(manual_clock, require_transition=True)
    for _ in range(3):
        await _observe(mgr, det, manual_clock, STANDING)
        manual_clock.advance(0.2)
    for _ in range(20):  # fallen but below the gate's fallen_conf (0.60)
        await _observe(mgr, det, manual_clock, [(0, 0.45)])
        manual_clock.advance(0.3)
    assert repos.events.count() == 0


# === duplicate / stale frame handling (service loop) ========================
@pytest.mark.asyncio
async def test_duplicate_frames_do_not_advance_confirmation(tmp_path):
    evclk = ManualClock()
    svc = build_live_service(tmp_path, dets=[_det("fallen", 0.95)], event_clock=evclk,
                             fall_confirm_seconds=2.0, detect_max_frame_age_seconds=10.0)
    svc.camera.set_peek(_img(), seq=1, age=0.0)
    for _ in range(10):  # SAME capture sequence, no clock advance
        await svc._detect_and_observe_once()
    assert svc.detector.calls == 1                       # de-duplicated -> inferred once
    assert svc.state_machine.state == FallState.POSSIBLE_FALL
    assert svc.state_machine.state != FallState.CONFIRMED_FALL  # dupes never confirm


@pytest.mark.asyncio
async def test_stale_frames_do_not_advance_evidence(tmp_path):
    evclk = ManualClock()
    svc = build_live_service(tmp_path, dets=[_det("fallen", 0.95)], event_clock=evclk,
                             fall_confirm_seconds=2.0, detect_max_frame_age_seconds=1.0)
    svc.camera.set_peek(_img(), seq=1, age=0.0)
    await svc._detect_and_observe_once()                 # fresh -> POSSIBLE
    assert svc.state_machine.state == FallState.POSSIBLE_FALL
    evclk.advance(2.1)                                   # enough wall time to confirm...
    for i in range(2, 12):
        svc.camera.set_peek(_img(), seq=i, age=5.0)      # ...but every new frame is stale
        await svc._detect_and_observe_once()
    assert svc.detector.calls == 1                       # stale frames never inferred
    assert svc._frames_dropped_stale == 10
    assert svc.state_machine.state != FallState.CONFIRMED_FALL  # stale evidence cannot confirm


# === conservative detector box gates ========================================
def test_tiny_fallen_box_is_not_evidence(manual_clock):
    det = make_yolo_detector(manual_clock, require_transition=False)
    det.min_fallen_box_area_frac = 0.05
    dets = _frame(det, manual_clock, [(0, 0.9, (10.0, 10.0, 50.0, 90.0))])  # ~0.01 area frac
    ev, _ = detections_to_evidence(dets, FALL, 0.55)
    assert ev is False
    assert dets[0].class_name == "fallen_posture"
    assert dets[0].metadata.get("rejection") == "too_small"
    assert det.rejection_counts.get("too_small") == 1


def test_edge_clipped_fallen_is_handled_conservatively(manual_clock):
    det = make_yolo_detector(manual_clock, require_transition=False)
    det.reject_edge_clipped_fallen = True
    # Clipped at the LEFT edge (partial person) -> not fall evidence.
    dets = _frame(det, manual_clock, [(0, 0.9, (2.0, 100.0, 300.0, 400.0))])
    ev, _ = detections_to_evidence(dets, FALL, 0.55)
    assert ev is False
    assert dets[0].class_name == "fallen_posture"
    assert dets[0].metadata.get("rejection", "").startswith("edge_clipped")
    # Clipped only at the BOTTOM (a real fall lands low) -> still fall evidence.
    dets2 = _frame(det, manual_clock, [(0, 0.9, (120.0, 120.0, 520.0, 478.0))])
    ev2, _ = detections_to_evidence(dets2, FALL, 0.55)
    assert ev2 is True


def test_detection_geometry_metadata_present(manual_clock):
    det = make_yolo_detector(manual_clock, require_transition=False)
    md = _frame(det, manual_clock, [(0, 0.9, (64.0, 48.0, 320.0, 240.0))])[0].metadata
    assert md["bbox_norm"] == [0.1, 0.1, 0.5, 0.5]
    assert md["area_frac"] == pytest.approx(0.16, abs=1e-3)
    assert md["vertical_center"] == pytest.approx(0.3, abs=1e-3)
    assert isinstance(md["edges"], list)


# === state-machine reconfirm cooldown =======================================
def test_reconfirm_cooldown_suppresses_repeat_then_allows(manual_clock):
    c = itertools.count(1)
    sm = FallEventStateMachine(
        confirm_seconds=2.0, clear_seconds=3.0, cooldown_seconds=0.0,
        clock=manual_clock, uid_factory=lambda: f"e{next(c)}",
        reconfirm_cooldown_seconds=20.0,
    )

    def confirmed(transitions):
        return any(t.to_state == FallState.CONFIRMED_FALL for t in transitions)

    # First confirmation.
    sm.observe(True, 0.9)
    manual_clock.advance(2.05)
    assert confirmed(sm.observe(True, 0.9))

    # Clear -> resolve.
    sm.observe(False)
    manual_clock.advance(3.1)
    sm.observe(False)
    sm.observe(False)

    # Immediate re-fall within the reconfirm window: must NOT confirm again.
    sm.observe(True, 0.9)
    manual_clock.advance(2.05)
    assert not confirmed(sm.observe(True, 0.9))

    # After the window elapses, a genuine fall confirms again.
    sm.observe(False)
    manual_clock.advance(20.0)
    sm.observe(True, 0.9)
    manual_clock.advance(2.05)
    assert confirmed(sm.observe(True, 0.9))


def test_reconfirm_cooldown_off_by_default_preserves_behaviour(manual_clock):
    c = itertools.count(1)
    sm = FallEventStateMachine(
        confirm_seconds=2.0, clear_seconds=3.0, cooldown_seconds=0.0,
        clock=manual_clock, uid_factory=lambda: f"e{next(c)}",
    )
    assert sm.reconfirm_cooldown_seconds == 0.0
    sm.observe(True, 0.9)
    manual_clock.advance(2.05)
    assert any(t.to_state == FallState.CONFIRMED_FALL for t in sm.observe(True, 0.9))
