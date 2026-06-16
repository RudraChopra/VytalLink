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
