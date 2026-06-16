"""Tests for YoloFallDetector using a fake model (no GPU / real weights)."""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import Frame
from vytallink.vision.detector_base import detections_to_evidence
from vytallink.vision.detector_yolo import FALLEN_POSTURE_CLASS, YoloFallDetector
from tests._fakes import FakeYoloModel, make_yolo_detector

FALL_CLASSES = {"fall", "fallen", "lying", "fall_detected", "person_fall"}


def _frame(i: int, clock: ManualClock) -> Frame:
    return Frame(frame_id=i, timestamp=clock.now(), source_id="cam", width=640, height=480, image=object())


def _infer(det, clock, script_frame, *, advance=0.1):
    det._model.set_script([script_frame])
    det._model._idx = 0
    dets = det.infer(_frame(det.inference_count + 1, clock))
    clock.advance(advance)
    return dets


def test_class_names_mapped():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)
    det._model.set_script([[(0, 0.9), (2, 0.7)]])
    dets = det.infer(_frame(1, clock))
    names = sorted(d.class_name for d in dets)
    assert names == ["fallen", "standing"]
    assert all(d.metadata["simulated"] is False for d in dets)


def test_require_transition_false_fallen_is_evidence():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)
    det._model.set_script([[(0, 0.91)]])
    dets = det.infer(_frame(1, clock))
    assert dets[0].class_name == "fallen"
    evidence, conf = detections_to_evidence(dets, FALL_CLASSES, 0.55)
    assert evidence is True and conf == pytest.approx(0.91, abs=1e-3)


def test_require_transition_true_already_lying_not_evidence():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=True)
    # Person already on the floor: fallen every frame, no preceding upright.
    last = []
    for _ in range(6):
        last = _infer(det, clock, [(0, 0.9)])
    assert last[0].class_name == FALLEN_POSTURE_CLASS
    evidence, _ = detections_to_evidence(last, FALL_CLASSES, 0.55)
    assert evidence is False


def test_require_transition_true_real_fall_becomes_evidence():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=True)
    for _ in range(3):  # establish upright
        _infer(det, clock, [(2, 0.9)])
    dets = []
    for _ in range(3):  # fall
        dets = _infer(det, clock, [(0, 0.85)])
    assert dets[0].class_name == "fallen"
    evidence, conf = detections_to_evidence(dets, FALL_CLASSES, 0.55)
    assert evidence is True


def test_no_image_returns_empty():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)
    f = Frame(frame_id=1, timestamp=clock.now(), source_id="cam")  # image=None
    assert det.infer(f) == []


def test_metrics_and_health_no_path_leak():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)
    for _ in range(3):
        _infer(det, clock, [(2, 0.8)])
    h = det.health()
    assert det.inference_count == 3
    assert h["inference_count"] == 3
    assert h["last_inference_ms"] is not None
    assert h["device"] == "cpu"
    assert h["classes"] == ["fallen", "sitting", "standing"]
    # CRITICAL: the absolute model path must NOT appear anywhere in health.
    assert h["model_file"] == "fall_detection.pt"
    assert "/fake/models" not in str(h)
    assert "model_path" not in h


def test_inference_failure_is_isolated():
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)

    def boom(image, **kw):
        raise RuntimeError("cuda blew up")

    det._model.predict = boom
    # Must not raise; returns no detections and records the (sanitized) error.
    assert det.infer(_frame(1, clock)) == []
    assert "cuda blew up" in (det.health()["last_error"] or "")


def test_health_degrades_when_inference_fails_then_recovers():
    """A model that loaded but fails every inference must NOT report status=ok."""
    clock = ManualClock()
    det = make_yolo_detector(clock, require_transition=False)
    assert det.health()["status"] == "ok"  # loaded, no inference yet

    def boom(image, **kw):
        raise RuntimeError("GPU fell off the bus")

    det._model.predict = boom
    det.infer(_frame(1, clock))
    assert det.last_infer_ok is False
    assert det.health()["status"] == "degraded"  # loaded but inference failing

    # A subsequent successful inference restores OK and clears the error.
    det._model = FakeYoloModel()
    det._model.set_script([[(2, 0.9)]])
    det.infer(_frame(2, clock))
    assert det.last_infer_ok is True
    assert det.health()["status"] == "ok"
    assert det.health()["last_error"] is None
