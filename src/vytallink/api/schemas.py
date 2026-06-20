"""Pydantic request models and response serializers for the API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vytallink.database.models import AlertRow, DeviceRow, EventRow, IncidentVitalRow, VitalRow
from vytallink.events.states import HumanLabel


def _finite_in_range(v: float | None, lo: float, hi: float, name: str) -> float | None:
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be a finite number")
    if not (lo <= v <= hi):
        raise ValueError(f"{name} out of plausible range [{lo}, {hi}]")
    return v


# Conservative, documented alias map (canonical -> accepted input keys). NOT an
# unbounded permissive parser: unknown keys are ignored, conflicting aliases with
# different values are rejected, and units are never guessed.
VITALS_ALIASES: dict[str, list[str]] = {
    "heart_rate": ["heart_rate", "hr", "bpm"],
    "respiratory_rate": ["respiratory_rate", "rr", "breathing_rate", "resp_rate", "br"],
    "motion": ["motion", "activity", "activity_level"],
    "posture": ["posture"],
    "battery": ["battery"],
    "phone_alert_score": ["phone_alert_score", "alert_score"],
    "device_id": ["device_id"],
    "timestamp": ["timestamp", "time", "ts", "source_timestamp", "device_timestamp", "recorded_at"],
    "sample_id": ["sample_id", "id"],
}
_NUMERIC_FIELDS = ("heart_rate", "respiratory_rate", "motion", "battery", "phone_alert_score")


class VitalsIngest(BaseModel):
    """iPhone vitals ingestion payload for POST /api/vitals.

    IMPORTANT: no prior iPhone contract existed in this repo — this schema is
    defined by VytalLink and must be verified against the real device. All fields
    are optional but at least one vital signal is required. A small, documented
    set of aliases is normalized; conflicting aliases with different values are
    rejected. Ranges are plausibility guards, NOT medical limits.
    """

    model_config = ConfigDict(extra="ignore")

    heart_rate: float | None = None
    respiratory_rate: float | None = None
    motion: float | None = None
    posture: str | None = Field(default=None, max_length=32)
    battery: float | None = None
    phone_alert_score: float | None = None
    device_id: str | None = Field(default=None, max_length=64)
    timestamp: str | None = None
    sample_id: str | None = Field(default=None, max_length=128)
    #: Which input key fed each canonical field (safe metadata, no values).
    accepted_aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out: dict[str, Any] = {}
        used: dict[str, str] = {}
        for canon, keys in VITALS_ALIASES.items():
            present = [(k, data[k]) for k in keys if k in data and data[k] is not None]
            distinct = {repr(v) for _, v in present}
            if len(distinct) > 1:
                raise ValueError(
                    f"conflicting values for {canon} via aliases {[k for k, _ in present]}"
                )
            if present:
                out[canon] = present[0][1]
                used[canon] = present[0][0]
        # Reject booleans-as-numbers (bool is an int subclass and would coerce).
        for f in _NUMERIC_FIELDS:
            if isinstance(out.get(f), bool):
                raise ValueError(f"{f} must be a number, not a boolean")
        out["accepted_aliases"] = used
        return out

    @property
    def contract_form(self) -> str:
        """'canonical' if only canonical names were used, else 'alias'."""
        return "canonical" if all(k == v for k, v in self.accepted_aliases.items()) else "alias"

    @field_validator("heart_rate")
    @classmethod
    def _hr(cls, v):  # noqa: ANN001
        return _finite_in_range(v, 20.0, 300.0, "heart_rate")

    @field_validator("respiratory_rate")
    @classmethod
    def _rr(cls, v):  # noqa: ANN001
        return _finite_in_range(v, 3.0, 60.0, "respiratory_rate")

    @field_validator("motion")
    @classmethod
    def _motion(cls, v):  # noqa: ANN001
        return _finite_in_range(v, 0.0, 1.0, "motion")

    @field_validator("battery", "phone_alert_score")
    @classmethod
    def _unit(cls, v):  # noqa: ANN001
        return _finite_in_range(v, 0.0, 1.0, "value")

    @field_validator("posture")
    @classmethod
    def _posture(cls, v):  # noqa: ANN001
        return v.strip().lower()[:32] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> "VitalsIngest":
        if self.heart_rate is None and self.respiratory_rate is None and self.motion is None and self.posture is None:
            raise ValueError("at least one of heart_rate, respiratory_rate, motion, posture is required")
        return self


class LabelRequest(BaseModel):
    """Body for POST /api/events/{id}/label."""

    label: HumanLabel

    @field_validator("label", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class ResolveRequest(BaseModel):
    """Body for POST /api/events/{id}/resolve."""

    note: str | None = Field(default=None, max_length=1000)


# --- serializers ----------------------------------------------------------
def event_to_dict(ev: EventRow, alerts: list[AlertRow] | None = None) -> dict[str, Any]:
    data = {
        "event_uid": ev.event_uid,
        "event_type": ev.event_type,
        "state": ev.state,
        "start_time": ev.start_time,
        "confirmed_time": ev.confirmed_time,
        "end_time": ev.end_time,
        "resolved_time": ev.resolved_time,
        "highest_confidence": ev.highest_confidence,
        "detection_count": ev.detection_count,
        "source_device": ev.source_device,
        "human_label": ev.human_label,
        "resolution_note": ev.resolution_note,
        "snapshot_path": ev.snapshot_path,
        "clip_path": ev.clip_path,
        "created_at": ev.created_at,
        "updated_at": ev.updated_at,
    }
    if alerts is not None:
        data["alerts"] = [alert_to_dict(a) for a in alerts]
        data["alert_delivered"] = any(a.success for a in alerts)
        data["alert_count"] = len(alerts)
    return data


def alert_to_dict(a: AlertRow) -> dict[str, Any]:
    return {
        "provider": a.provider,
        "attempt_time": a.attempt_time,
        "success": a.success,
        "failure_message": a.failure_message,
        "response_metadata": a.response_metadata,
    }


def vital_to_dict(v: VitalRow) -> dict[str, Any]:
    md = v.metadata or {}
    # Legacy top-level fields are preserved unchanged; new fields (from the iPhone
    # payload, stored in metadata) are added without breaking existing clients.
    return {
        "timestamp": v.timestamp,
        "device_id": v.device_id,
        "heart_rate": v.heart_rate,
        "motion": v.motion,
        "connection_quality": v.connection_quality,
        "battery": v.battery,
        "simulated": v.simulated,
        "respiratory_rate": md.get("respiratory_rate"),
        "posture": md.get("posture"),
        "phone_alert_score": md.get("phone_alert_score"),
        "source": md.get("source"),
    }


def incident_vital_to_dict(s: IncidentVitalRow) -> dict[str, Any]:
    """Safe serializer for an incident vitals snapshot (no credentials/raw bodies)."""
    return {
        "event_uid": s.event_uid,
        "camera_id": s.camera_id,
        "confirmed_time": s.confirmed_time,
        "vitals_sample_id": s.vitals_sample_id,
        "heart_rate": s.heart_rate,
        "respiratory_rate": s.respiratory_rate,
        "posture": s.posture,
        "phone_alert_score": s.phone_alert_score,
        "computed_alert_level": s.computed_alert_level,
        "computed_alert_score": s.computed_alert_score,
        "reason_codes": s.reason_codes,
        "source_timestamp": s.source_timestamp,
        "received_at": s.received_at,
        "vitals_age_seconds": s.vitals_age_seconds,
        "vitals_freshness": s.vitals_freshness,
        "vitals_available": s.vitals_available,
        "vitals_source": s.vitals_source,
        "synthetic": s.synthetic,
        "snapshot_version": s.snapshot_version,
        "created_at": s.created_at,
    }


def device_to_dict(d: DeviceRow) -> dict[str, Any]:
    return {
        "device_id": d.device_id,
        "device_type": d.device_type,
        "display_name": d.display_name,
        "connection_status": d.connection_status,
        "last_seen": d.last_seen,
        "last_error": d.last_error,
        "metadata": d.metadata,
        "updated_at": d.updated_at,
    }
