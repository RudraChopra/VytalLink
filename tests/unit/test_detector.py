"""Tests for the simulated fall detector + evidence mapping."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vytallink.common.types import Frame, RawDetection
from vytallink.vision.detector_base import detections_to_evidence
from vytallink.vision.detector_simulated import Scenario, SimulatedFallDetector


def _frame(frame_id: int = 1) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        source_id="camera-1",
        width=640,
        height=480,
    )


def test_normal_scenario_no_fall_evidence():
    det = SimulatedFallDetector(scenario=Scenario.NORMAL)
    det.load()
    dets = det.infer(_frame())
    assert len(dets) == 1
    assert dets[0].class_name == "person"
    evidence, conf = detections_to_evidence(dets, {"fall"}, 0.55)
    assert evidence is False
    assert conf == 0.0


def test_fall_scenario_produces_evidence():
    det = SimulatedFallDetector(scenario=Scenario.FALL)
    det.load()
    dets = det.infer(_frame(frame_id=3))
    assert dets[0].class_name == "fall"
    assert dets[0].confidence >= 0.55
    assert dets[0].metadata["simulated"] is True
    evidence, conf = detections_to_evidence(dets, {"fall"}, 0.55)
    assert evidence is True
    assert conf >= 0.55


def test_evidence_requires_threshold():
    low = [
        RawDetection(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            class_id=1,
            class_name="fall",
            confidence=0.40,
            bbox=(0, 0, 1, 1),
            source_id="camera-1",
            frame_id=1,
        )
    ]
    evidence, conf = detections_to_evidence(low, {"fall"}, 0.55)
    assert evidence is False


def test_scripted_sequence():
    det = SimulatedFallDetector()
    det.load()
    det.set_script([True, False, True])
    assert det.infer(_frame(1))[0].class_name == "fall"
    assert det.infer(_frame(2))[0].class_name == "person"
    assert det.infer(_frame(3))[0].class_name == "fall"


def test_health_reports_scenario():
    det = SimulatedFallDetector(scenario=Scenario.FALL)
    det.load()
    h = det.health()
    assert h["scenario"] == "fall"
    assert h["simulated"] is True
    assert h["loaded"] is True


def test_confidence_is_deterministic():
    det1 = SimulatedFallDetector(scenario=Scenario.FALL)
    det2 = SimulatedFallDetector(scenario=Scenario.FALL)
    det1.load()
    det2.load()
    assert det1.infer(_frame(5))[0].confidence == det2.infer(_frame(5))[0].confidence
