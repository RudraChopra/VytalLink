"""Row dataclasses mapping to the SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {"value": data}
    except (json.JSONDecodeError, TypeError):
        return {}


def _dumps(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


@dataclass(slots=True)
class EventRow:
    event_uid: str
    state: str
    start_time: str
    source_device: str
    event_type: str = "fall"
    confirmed_time: str | None = None
    end_time: str | None = None
    resolved_time: str | None = None
    highest_confidence: float = 0.0
    detection_count: int = 0
    snapshot_path: str | None = None
    clip_path: str | None = None
    human_label: str | None = None
    resolution_note: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EventRow":
        return cls(
            id=row["id"],
            event_uid=row["event_uid"],
            event_type=row["event_type"],
            state=row["state"],
            start_time=row["start_time"],
            confirmed_time=row["confirmed_time"],
            end_time=row["end_time"],
            resolved_time=row["resolved_time"],
            highest_confidence=row["highest_confidence"],
            detection_count=row["detection_count"],
            source_device=row["source_device"],
            snapshot_path=row["snapshot_path"],
            clip_path=row["clip_path"],
            human_label=row["human_label"],
            resolution_note=row["resolution_note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VitalRow:
    timestamp: str
    device_id: str
    heart_rate: float | None = None
    motion: float | None = None
    connection_quality: float | None = None
    battery: float | None = None
    simulated: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "VitalRow":
        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            device_id=row["device_id"],
            heart_rate=row["heart_rate"],
            motion=row["motion"],
            connection_quality=row["connection_quality"],
            battery=row["battery"],
            simulated=bool(row["simulated"]),
            metadata=_loads(row["metadata"]),
            created_at=row["created_at"],
        )

    def metadata_json(self) -> str | None:
        return _dumps(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlertRow:
    event_uid: str
    provider: str
    attempt_time: str
    success: bool = False
    failure_message: str | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AlertRow":
        return cls(
            id=row["id"],
            event_uid=row["event_uid"],
            provider=row["provider"],
            attempt_time=row["attempt_time"],
            success=bool(row["success"]),
            failure_message=row["failure_message"],
            response_metadata=_loads(row["response_metadata"]),
            created_at=row["created_at"],
        )

    def response_json(self) -> str | None:
        return _dumps(self.response_metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IncidentVitalRow:
    """One vitals snapshot tied to a confirmed fall incident (event_uid unique).

    Contains only safe, normalized fields — never credentials, RTSP URLs, or raw
    request bodies. ``reason_codes`` is a JSON-encoded list of alert reason codes.
    """

    event_uid: str
    camera_id: str
    confirmed_time: str | None = None
    vitals_sample_id: str | None = None
    heart_rate: float | None = None
    respiratory_rate: float | None = None
    posture: str | None = None
    phone_alert_score: float | None = None
    computed_alert_level: str | None = None
    computed_alert_score: int | None = None
    reason_codes: list[str] = field(default_factory=list)
    source_timestamp: str | None = None
    received_at: str | None = None
    vitals_age_seconds: float | None = None
    vitals_freshness: str | None = None
    vitals_available: bool = False
    vitals_source: str | None = None
    synthetic: bool = False
    snapshot_version: int = 1
    created_at: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "IncidentVitalRow":
        raw = row["reason_codes"]
        try:
            codes = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            codes = []
        return cls(
            id=row["id"], event_uid=row["event_uid"], camera_id=row["camera_id"],
            confirmed_time=row["confirmed_time"], vitals_sample_id=row["vitals_sample_id"],
            heart_rate=row["heart_rate"], respiratory_rate=row["respiratory_rate"],
            posture=row["posture"], phone_alert_score=row["phone_alert_score"],
            computed_alert_level=row["computed_alert_level"],
            computed_alert_score=row["computed_alert_score"],
            reason_codes=codes if isinstance(codes, list) else [],
            source_timestamp=row["source_timestamp"], received_at=row["received_at"],
            vitals_age_seconds=row["vitals_age_seconds"], vitals_freshness=row["vitals_freshness"],
            vitals_available=bool(row["vitals_available"]), vitals_source=row["vitals_source"],
            synthetic=bool(row["synthetic"]), snapshot_version=row["snapshot_version"],
            created_at=row["created_at"],
        )

    def reason_codes_json(self) -> str:
        return json.dumps(list(self.reason_codes), separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeviceRow:
    device_id: str
    device_type: str
    display_name: str = ""
    connection_status: str = "unknown"
    last_seen: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DeviceRow":
        return cls(
            device_id=row["device_id"],
            device_type=row["device_type"],
            display_name=row["display_name"],
            connection_status=row["connection_status"],
            last_seen=row["last_seen"],
            last_error=row["last_error"],
            metadata=_loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def metadata_json(self) -> str | None:
        return _dumps(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
