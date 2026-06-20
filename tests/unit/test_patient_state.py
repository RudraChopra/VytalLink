"""Deterministic tests for freshness, alert scoring, and patient-state
aggregation (pure functions, injected ``now`` — no sleeps, no hardware)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from vytallink.monitoring.alert_score import (
    R_FALL_CONFIRMED,
    R_FALL_SUSPECTED,
    R_HR_HIGH,
    R_INCIDENT_ACTIVE,
    R_PERSON_AMBIGUOUS,
    R_VISION_UNAVAILABLE,
    R_VITALS_STALE,
    R_VITALS_UNAVAILABLE,
    ScoreThresholds,
    score_patient,
)
from vytallink.monitoring.freshness import (
    FreshnessThresholds,
    camera_freshness,
    vitals_freshness,
)
from vytallink.monitoring.patient_state import build_patient_state

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FT = FreshnessThresholds()
ST = ScoreThresholds()


def _vital(hr=72.0, *, rr=None, posture=None, motion=0.1, simulated=False, source="iphone"):
    return SimpleNamespace(
        heart_rate=hr, motion=motion, device_id="iphone-1", simulated=simulated,
        metadata={"respiratory_rate": rr, "posture": posture, "source": source},
    )


def _cam(cid, *, connected=True, fall_state="normal", age_s=1.0, person_count=1, conf=None):
    return {"id": cid, "connected": connected, "fall_state": fall_state,
            "frame_age_seconds": age_s, "person_count": person_count, "fall_confidence": conf}


# --- freshness -------------------------------------------------------------
def test_vitals_freshness_bands():
    assert vitals_freshness(2, FT) == "fresh"
    assert vitals_freshness(20, FT) == "aging"
    assert vitals_freshness(60, FT) == "stale"
    assert vitals_freshness(200, FT) == "unavailable"   # older than vitals_stale (90)
    assert vitals_freshness(None, FT) == "unavailable"


def test_camera_freshness_states():
    assert camera_freshness(2, connected=True, t=FT) == "fresh"
    assert camera_freshness(10, connected=True, t=FT) == "stale"
    assert camera_freshness(None, connected=False, t=FT) == "offline"
    assert camera_freshness(None, connected=True, t=FT) == "unavailable"


# --- scoring rules ---------------------------------------------------------
def _score(**over):
    base = dict(vision_overall_state="normal", vision_available=True, any_camera_fresh=True,
                person_ambiguous=False, incident_active=False, heart_rate=72.0,
                respiratory_rate=None, vitals_usable=True, vitals_freshness="fresh", thr=ST)
    base.update(over)
    return score_patient(**base)


def test_confirmed_fall_is_critical_even_with_normal_vitals():
    s = _score(vision_overall_state="confirmed_fall", heart_rate=72.0)
    assert s["level"] == "critical"
    assert R_FALL_CONFIRMED in s["reasons"]


def test_suspected_fall_is_warning():
    s = _score(vision_overall_state="possible_fall")
    assert s["level"] == "warning"
    assert R_FALL_SUSPECTED in s["reasons"]


def test_abnormal_hr_without_fall_is_warning():
    s = _score(heart_rate=180.0)
    assert s["level"] == "warning"
    assert R_HR_HIGH in s["reasons"]
    assert R_FALL_CONFIRMED not in s["reasons"]


def test_stale_vitals_are_not_reassuring_and_hr_not_evaluated():
    # Stale vitals: the out-of-range HR must NOT be read as a warning (not usable),
    # but staleness must be flagged — never treated as "normal/ok".
    s = _score(heart_rate=180.0, vitals_usable=False, vitals_freshness="stale")
    assert R_HR_HIGH not in s["reasons"]
    assert R_VITALS_STALE in s["reasons"]


def test_unavailable_vitals_flagged():
    s = _score(heart_rate=None, vitals_usable=False, vitals_freshness="unavailable")
    assert R_VITALS_UNAVAILABLE in s["reasons"]


def test_vision_unavailable_when_no_fresh_camera():
    s = _score(any_camera_fresh=False)
    assert R_VISION_UNAVAILABLE in s["reasons"]


def test_person_ambiguous_and_incident_active_reasons():
    s = _score(person_ambiguous=True, incident_active=True)
    assert R_PERSON_AMBIGUOUS in s["reasons"]
    assert R_INCIDENT_ACTIVE in s["reasons"]


# --- aggregation -----------------------------------------------------------
def _state(cameras, vital=None, incident=None):
    return build_patient_state(now=NOW, vital=vital, received_at=NOW,
                               source_timestamp=NOW - timedelta(seconds=2), cameras=cameras,
                               active_incident_id=incident, fresh_thr=FT, score_thr=ST)


def test_worst_of_fresh_cameras_with_source_id():
    ps = _state([_cam("camera_1", fall_state="confirmed_fall"), _cam("camera_2", fall_state="normal")])
    assert ps["vision"]["overall_state"] == "confirmed_fall"
    assert ps["vision"]["source_camera_id"] == "camera_1"


def test_offline_camera_does_not_erase_other_evidence():
    ps = _state([_cam("camera_1", connected=False, age_s=None, fall_state="normal"),
                 _cam("camera_2", fall_state="confirmed_fall")])
    assert ps["vision"]["overall_state"] == "confirmed_fall"   # fresh camera_2 still counts
    assert ps["vision"]["source_camera_id"] == "camera_2"
    assert ps["vision"]["cameras"]["camera_1"]["freshness"] == "offline"


def test_both_cameras_offline_is_vision_unavailable():
    ps = _state([_cam("camera_1", connected=False, age_s=None),
                 _cam("camera_2", connected=False, age_s=None)])
    assert ps["freshness"]["vision"] == "offline"
    assert R_VISION_UNAVAILABLE in ps["alert"]["reasons"]


def test_no_cameras_and_no_vitals():
    ps = _state([], vital=None)
    assert ps["vision"]["overall_state"] == "normal"
    assert ps["vitals"]["freshness"] == "unavailable"
    assert R_VISION_UNAVAILABLE in ps["alert"]["reasons"]


def test_person_count_ambiguous_when_fresh_cameras_disagree():
    ps = _state([_cam("camera_1", person_count=1), _cam("camera_2", person_count=2)])
    assert ps["vision"]["person_count_ambiguous"] is True
    assert R_PERSON_AMBIGUOUS in ps["alert"]["reasons"]


def test_confidences_are_not_summed():
    # Two cameras each with confidence 0.5 must not produce a combined 1.0.
    ps = _state([_cam("camera_1", conf=0.5, fall_state="confirmed_fall"),
                 _cam("camera_2", conf=0.5, fall_state="confirmed_fall")])
    confs = [c["fall_confidence"] for c in ps["vision"]["cameras"].values()]
    assert confs == [0.5, 0.5]                  # preserved per camera, never summed
    assert "score" in ps["alert"] and ps["alert"]["score"] <= 3


def test_raw_and_computed_are_distinguishable():
    ps = _state([_cam("camera_1")], vital=_vital(hr=72.0, rr=16.0, posture="upright"))
    # raw vitals preserved verbatim
    assert ps["vitals"]["heart_rate"] == 72.0
    assert ps["vitals"]["respiratory_rate"] == 16.0
    assert ps["vitals"]["posture"] == "upright"
    assert ps["vitals"]["source"] == "iphone"
    # computed block carries the score + a non-diagnosis marker
    assert ps["not_a_diagnosis"] is True
    assert set(ps["alert"]) == {"level", "score", "reasons"}
