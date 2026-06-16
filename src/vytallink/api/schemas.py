"""Pydantic request models and response serializers for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from vytallink.database.models import AlertRow, DeviceRow, EventRow, VitalRow
from vytallink.events.states import HumanLabel


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
    return {
        "timestamp": v.timestamp,
        "device_id": v.device_id,
        "heart_rate": v.heart_rate,
        "motion": v.motion,
        "connection_quality": v.connection_quality,
        "battery": v.battery,
        "simulated": v.simulated,
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
