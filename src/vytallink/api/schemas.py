"""Pydantic request models and response serializers for the API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from vytallink.database.models import AlertRow, DeviceRow, EventRow, VitalRow
from vytallink.events.states import HumanLabel


def _finite_in_range(v: float | None, lo: float, hi: float, name: str) -> float | None:
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be a finite number")
    if not (lo <= v <= hi):
        raise ValueError(f"{name} out of plausible range [{lo}, {hi}]")
    return v


class VitalsIngest(BaseModel):
    """iPhone vitals ingestion payload for POST /api/vitals.

    IMPORTANT: no prior iPhone contract existed in this repo — this schema is
    defined by VytalLink and must be verified against the real device. All fields
    are optional but at least one vital signal is required. Common field-name
    variants are accepted. Ranges are plausibility guards, NOT medical limits.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    heart_rate: float | None = Field(default=None, validation_alias=AliasChoices("heart_rate", "hr", "bpm"))
    respiratory_rate: float | None = Field(
        default=None, validation_alias=AliasChoices("respiratory_rate", "rr", "breathing_rate", "resp_rate")
    )
    motion: float | None = Field(default=None, validation_alias=AliasChoices("motion", "activity", "activity_level"))
    posture: str | None = Field(default=None, max_length=32)
    battery: float | None = None
    phone_alert_score: float | None = Field(
        default=None, validation_alias=AliasChoices("phone_alert_score", "alert_score")
    )
    device_id: str | None = Field(default=None, max_length=64)
    timestamp: str | None = Field(default=None, validation_alias=AliasChoices("timestamp", "time", "ts"))
    sample_id: str | None = Field(default=None, max_length=128, validation_alias=AliasChoices("sample_id", "id"))

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
