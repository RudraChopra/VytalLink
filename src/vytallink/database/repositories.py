"""Data-access repositories. All SQL is parameterized.

Column names used in dynamic UPDATEs are validated against per-table
whitelists so a caller can never inject SQL through a field name.
"""

from __future__ import annotations

from typing import Any

from vytallink.common.errors import DatabaseError, NotFoundError
from vytallink.database.db import Database
from vytallink.database.models import AlertRow, DeviceRow, EventRow, VitalRow


class EventRepository:
    _UPDATABLE = frozenset(
        {
            "state",
            "confirmed_time",
            "end_time",
            "resolved_time",
            "highest_confidence",
            "detection_count",
            "snapshot_path",
            "clip_path",
            "human_label",
            "resolution_note",
        }
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, event: EventRow) -> EventRow:
        now = self.db.now_iso()
        event.created_at = event.created_at or now
        event.updated_at = now
        cur = self.db.execute(
            """
            INSERT INTO events (
                event_uid, event_type, state, start_time, confirmed_time,
                end_time, resolved_time, highest_confidence, detection_count,
                source_device, snapshot_path, clip_path, human_label,
                resolution_note, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.event_uid,
                event.event_type,
                event.state,
                event.start_time,
                event.confirmed_time,
                event.end_time,
                event.resolved_time,
                event.highest_confidence,
                event.detection_count,
                event.source_device,
                event.snapshot_path,
                event.clip_path,
                event.human_label,
                event.resolution_note,
                event.created_at,
                event.updated_at,
            ),
        )
        event.id = cur.lastrowid
        return event

    def get_by_uid(self, event_uid: str) -> EventRow | None:
        row = self.db.query_one(
            "SELECT * FROM events WHERE event_uid = ?", (event_uid,)
        )
        return EventRow.from_row(row) if row else None

    def require(self, event_uid: str) -> EventRow:
        ev = self.get_by_uid(event_uid)
        if ev is None:
            raise NotFoundError(f"Event not found: {event_uid}")
        return ev

    def update(self, event_uid: str, **fields: Any) -> EventRow:
        bad = set(fields) - self._UPDATABLE
        if bad:
            raise DatabaseError(f"Cannot update non-updatable event columns: {sorted(bad)}")
        if not fields:
            return self.require(event_uid)
        assignments = ", ".join(f"{col} = ?" for col in fields)
        params = list(fields.values()) + [self.db.now_iso(), event_uid]
        cur = self.db.execute(
            f"UPDATE events SET {assignments}, updated_at = ? WHERE event_uid = ?",
            params,
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"Event not found: {event_uid}")
        return self.require(event_uid)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        state: str | None = None,
        human_label: str | None = None,
    ) -> list[EventRow]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        clauses: list[str] = []
        params: list[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if human_label:
            clauses.append("human_label = ?")
            params.append(human_label)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.db.query_all(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        )
        return [EventRow.from_row(r) for r in rows]

    def count(self, *, state: str | None = None) -> int:
        if state:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM events WHERE state = ?", (state,))
        else:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM events")
        return int(row["n"]) if row else 0


class VitalRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, vital: VitalRow) -> VitalRow:
        now = self.db.now_iso()
        vital.created_at = vital.created_at or now
        cur = self.db.execute(
            """
            INSERT INTO vitals (
                timestamp, device_id, heart_rate, motion, connection_quality,
                battery, simulated, metadata, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                vital.timestamp,
                vital.device_id,
                vital.heart_rate,
                vital.motion,
                vital.connection_quality,
                vital.battery,
                1 if vital.simulated else 0,
                vital.metadata_json(),
                vital.created_at,
            ),
        )
        vital.id = cur.lastrowid
        return vital

    def latest(self, device_id: str | None = None) -> VitalRow | None:
        if device_id:
            row = self.db.query_one(
                "SELECT * FROM vitals WHERE device_id = ? ORDER BY id DESC LIMIT 1",
                (device_id,),
            )
        else:
            row = self.db.query_one("SELECT * FROM vitals ORDER BY id DESC LIMIT 1")
        return VitalRow.from_row(row) if row else None

    def list(
        self, *, limit: int = 50, offset: int = 0, device_id: str | None = None
    ) -> list[VitalRow]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        if device_id:
            rows = self.db.query_all(
                "SELECT * FROM vitals WHERE device_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (device_id, limit, offset),
            )
        else:
            rows = self.db.query_all(
                "SELECT * FROM vitals ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
            )
        return [VitalRow.from_row(r) for r in rows]

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM vitals")
        return int(row["n"]) if row else 0


class AlertRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, alert: AlertRow) -> AlertRow:
        now = self.db.now_iso()
        alert.created_at = alert.created_at or now
        cur = self.db.execute(
            """
            INSERT INTO alerts (
                event_uid, provider, attempt_time, success,
                failure_message, response_metadata, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                alert.event_uid,
                alert.provider,
                alert.attempt_time,
                1 if alert.success else 0,
                alert.failure_message,
                alert.response_json(),
                alert.created_at,
            ),
        )
        alert.id = cur.lastrowid
        return alert

    def list_for_event(self, event_uid: str) -> list[AlertRow]:
        rows = self.db.query_all(
            "SELECT * FROM alerts WHERE event_uid = ? ORDER BY id ASC", (event_uid,)
        )
        return [AlertRow.from_row(r) for r in rows]

    def list(self, *, limit: int = 50, offset: int = 0) -> list[AlertRow]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        rows = self.db.query_all(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        )
        return [AlertRow.from_row(r) for r in rows]

    def count(self, *, event_uid: str | None = None, success: bool | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if event_uid is not None:
            clauses.append("event_uid = ?")
            params.append(event_uid)
        if success is not None:
            clauses.append("success = ?")
            params.append(1 if success else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.db.query_one(f"SELECT COUNT(*) AS n FROM alerts {where}", params)
        return int(row["n"]) if row else 0


class DeviceRepository:
    _UPDATABLE = frozenset(
        {"display_name", "connection_status", "last_seen", "last_error", "metadata"}
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, device: DeviceRow) -> DeviceRow:
        now = self.db.now_iso()
        existing = self.get(device.device_id)
        if existing is None:
            device.created_at = now
            device.updated_at = now
            self.db.execute(
                """
                INSERT INTO devices (
                    device_id, device_type, display_name, connection_status,
                    last_seen, last_error, metadata, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    device.device_id,
                    device.device_type,
                    device.display_name,
                    device.connection_status,
                    device.last_seen,
                    device.last_error,
                    device.metadata_json(),
                    device.created_at,
                    device.updated_at,
                ),
            )
            return device
        return self.update(
            device.device_id,
            display_name=device.display_name,
            connection_status=device.connection_status,
            last_seen=device.last_seen,
            last_error=device.last_error,
            metadata=device.metadata_json(),
        )

    def update(self, device_id: str, **fields: Any) -> DeviceRow:
        bad = set(fields) - self._UPDATABLE
        if bad:
            raise DatabaseError(f"Cannot update non-updatable device columns: {sorted(bad)}")
        if not fields:
            return self.require(device_id)
        assignments = ", ".join(f"{col} = ?" for col in fields)
        params = list(fields.values()) + [self.db.now_iso(), device_id]
        cur = self.db.execute(
            f"UPDATE devices SET {assignments}, updated_at = ? WHERE device_id = ?",
            params,
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"Device not found: {device_id}")
        return self.require(device_id)

    def get(self, device_id: str) -> DeviceRow | None:
        row = self.db.query_one("SELECT * FROM devices WHERE device_id = ?", (device_id,))
        return DeviceRow.from_row(row) if row else None

    def require(self, device_id: str) -> DeviceRow:
        d = self.get(device_id)
        if d is None:
            raise NotFoundError(f"Device not found: {device_id}")
        return d

    def list(self) -> list[DeviceRow]:
        rows = self.db.query_all("SELECT * FROM devices ORDER BY device_type, device_id")
        return [DeviceRow.from_row(r) for r in rows]


class Repositories:
    """Convenience bundle of all repositories over one database."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.events = EventRepository(db)
        self.vitals = VitalRepository(db)
        self.alerts = AlertRepository(db)
        self.devices = DeviceRepository(db)
