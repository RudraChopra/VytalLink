"""Normalized patient state: combine latest vitals + per-camera fall state into
one explainable structure, keeping RAW source data and COMPUTED aggregates
distinguishable.

Pure and deterministic (``now`` injected). Confidence values from different
cameras are never combined; the aggregate vision state is the worst-of the
*fresh* cameras only, with the responsible camera id surfaced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from vytallink.common.clock import isoformat
from vytallink.monitoring.alert_score import ScoreThresholds, score_patient
from vytallink.monitoring.freshness import (
    FreshnessThresholds,
    camera_freshness,
    camera_is_fresh,
    vitals_freshness,
    vitals_is_usable,
)

#: Fall-state severity (low -> high) for worst-of aggregation across cameras.
_FALL_RANK = {"normal": 0, "resolved": 1, "possible_fall": 2, "recovering": 3, "confirmed_fall": 4}


def _age_seconds(now: datetime, ts: datetime | None) -> float | None:
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds())


def _aggregate_cameras(cameras: list[dict[str, Any]], t: FreshnessThresholds) -> dict[str, Any]:
    """Worst-of FRESH cameras -> overall vision state + responsible camera.

    cameras: list of {id, connected, fall_state, frame_age_seconds, person_count}.
    """
    per_cam: dict[str, Any] = {}
    fresh_states: list[tuple[str, str]] = []  # (camera_id, fall_state) for fresh cams
    fresh_person_counts: list[int] = []
    any_fresh = False
    for c in cameras:
        cid = c["id"]
        fresh = camera_freshness(c.get("frame_age_seconds"), connected=bool(c.get("connected")), t=t)
        is_fresh = camera_is_fresh(fresh)
        any_fresh = any_fresh or is_fresh
        fall_state = c.get("fall_state", "normal")
        pc = c.get("person_count")
        per_cam[cid] = {
            "connected": bool(c.get("connected")),
            "freshness": fresh,
            "frame_age_ms": round(c["frame_age_seconds"] * 1000.0, 1) if c.get("frame_age_seconds") is not None else None,
            "fall_state": fall_state,
            "person_count": pc,
            "fall_confidence": c.get("fall_confidence"),
        }
        if is_fresh:
            fresh_states.append((cid, fall_state))
            if isinstance(pc, int):
                fresh_person_counts.append(pc)

    # Overall = worst-of fresh cameras (an offline/stale camera cannot mask a
    # fresh camera's fall, nor erase another camera's evidence).
    overall_state = "normal"
    source_camera_id = None
    best_rank = -1
    for cid, st in fresh_states:
        r = _FALL_RANK.get(st, 0)
        if r > best_rank:
            best_rank, overall_state, source_camera_id = r, st, cid
    if best_rank <= 0:
        source_camera_id = None  # nothing notable to attribute

    # Ambiguous only when ≥2 fresh cameras each see people but DISAGREE on the
    # count. A camera seeing 0 (different field of view) is not a conflict.
    nonzero_counts = {p for p in fresh_person_counts if p > 0}
    return {
        "overall_state": overall_state,
        "source_camera_id": source_camera_id,
        "cameras": per_cam,
        "any_camera_fresh": any_fresh,
        "vision_available": len(cameras) > 0,
        "person_count_ambiguous": len(nonzero_counts) > 1,
        "person_count": (max(fresh_person_counts) if fresh_person_counts else None),
    }


def build_patient_state(
    *,
    now: datetime,
    vital: Any | None,            # VitalRow or None (raw latest vitals)
    received_at: datetime | None,  # server-received time of that vital
    source_timestamp: datetime | None,  # parsed source timestamp of that vital
    cameras: list[dict[str, Any]],
    active_incident_id: str | None,
    fresh_thr: FreshnessThresholds,
    score_thr: ScoreThresholds,
) -> dict[str, Any]:
    """Assemble the normalized patient state (raw vitals + computed aggregates)."""
    md = dict(getattr(vital, "metadata", {}) or {}) if vital is not None else {}
    hr = getattr(vital, "heart_rate", None) if vital is not None else None
    rr = md.get("respiratory_rate")
    # No vital -> unavailable, regardless of any stray timestamp.
    age = _age_seconds(now, source_timestamp) if vital is not None else None
    v_fresh = vitals_freshness(age, fresh_thr)
    usable = vitals_is_usable(v_fresh)

    # RAW vitals (clearly source data; never overwritten by computed values).
    vitals_block = {
        "heart_rate": hr,
        "respiratory_rate": rr,
        "motion": getattr(vital, "motion", None) if vital is not None else None,
        "posture": md.get("posture"),
        "phone_alert_score": md.get("phone_alert_score"),
        "device_id": getattr(vital, "device_id", None) if vital is not None else None,
        "source": md.get("source", "wearable" if vital is not None else None),
        "simulated": bool(getattr(vital, "simulated", False)) if vital is not None else None,
        "source_timestamp": isoformat(source_timestamp) if source_timestamp else None,
        "received_at": isoformat(received_at) if received_at else None,
        "age_seconds": round(age, 2) if age is not None else None,
        "freshness": v_fresh,
    }

    vision = _aggregate_cameras(cameras, fresh_thr)
    # Vision freshness summary: fresh if any fresh camera, else stale/offline/unavailable.
    if not vision["vision_available"]:
        vision_freshness = "unavailable"
    elif vision["any_camera_fresh"]:
        vision_freshness = "fresh"
    elif any(c["connected"] for c in vision["cameras"].values()):
        vision_freshness = "stale"
    else:
        vision_freshness = "offline"

    alert = score_patient(
        vision_overall_state=vision["overall_state"],
        vision_available=vision["vision_available"],
        any_camera_fresh=vision["any_camera_fresh"],
        person_ambiguous=vision["person_count_ambiguous"],
        incident_active=bool(active_incident_id),
        heart_rate=hr,
        respiratory_rate=rr,
        vitals_usable=usable,
        vitals_freshness=v_fresh,
        thr=score_thr,
    )

    return {
        "generated_at": isoformat(now),               # processing timestamp
        "not_a_diagnosis": True,                       # the score is informational only
        "vitals": vitals_block,                        # RAW
        "vision": {                                    # COMPUTED aggregate
            "overall_state": vision["overall_state"],
            "source_camera_id": vision["source_camera_id"],
            "active_incident_id": active_incident_id,
            "person_count": vision["person_count"],
            "person_count_ambiguous": vision["person_count_ambiguous"],
            "cameras": vision["cameras"],
        },
        "freshness": {
            "vitals": v_fresh,
            "vitals_age_seconds": vitals_block["age_seconds"],
            "vision": vision_freshness,
        },
        "alert": alert,                                # COMPUTED: level/score/reasons
    }
