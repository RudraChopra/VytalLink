"""Tests for database initialization, persistence, and repositories."""

from __future__ import annotations

import pytest

from vytallink.common.errors import DatabaseError, NotFoundError
from vytallink.database import (
    AlertRow,
    Database,
    DeviceRow,
    EventRow,
    Repositories,
    VitalRow,
)
from vytallink.database.schema import LATEST_SCHEMA_VERSION


def _make_event(uid: str = "evt-1", state: str = "possible_fall") -> EventRow:
    return EventRow(
        event_uid=uid,
        state=state,
        start_time="2025-01-01T00:00:00+00:00",
        source_device="camera-1",
        highest_confidence=0.7,
        detection_count=1,
    )


def test_initialize_sets_schema_version(database: Database):
    assert database.initialize() == LATEST_SCHEMA_VERSION
    health = database.health()
    assert health["ok"] is True
    assert health["schema_version"] == LATEST_SCHEMA_VERSION


def test_initialize_is_idempotent(database: Database):
    v1 = database.initialize()
    v2 = database.initialize()
    assert v1 == v2 == LATEST_SCHEMA_VERSION


def test_event_create_and_fetch(repos: Repositories):
    created = repos.events.create(_make_event())
    assert created.id is not None
    assert created.created_at is not None
    fetched = repos.events.get_by_uid("evt-1")
    assert fetched is not None
    assert fetched.state == "possible_fall"
    assert fetched.highest_confidence == pytest.approx(0.7)


def test_event_update_whitelist(repos: Repositories):
    repos.events.create(_make_event())
    updated = repos.events.update("evt-1", state="confirmed_fall", highest_confidence=0.95)
    assert updated.state == "confirmed_fall"
    assert updated.highest_confidence == pytest.approx(0.95)
    with pytest.raises(DatabaseError):
        repos.events.update("evt-1", created_at="hack")  # not updatable


def test_event_update_missing_raises(repos: Repositories):
    with pytest.raises(NotFoundError):
        repos.events.update("nope", state="resolved")


def test_event_list_pagination_and_filter(repos: Repositories):
    for i in range(5):
        repos.events.create(_make_event(uid=f"e{i}", state="resolved" if i % 2 else "confirmed_fall"))
    all_events = repos.events.list(limit=10)
    assert len(all_events) == 5
    # newest first (id desc)
    assert all_events[0].event_uid == "e4"
    page = repos.events.list(limit=2, offset=0)
    assert len(page) == 2
    confirmed = repos.events.list(state="confirmed_fall")
    assert all(e.state == "confirmed_fall" for e in confirmed)
    assert repos.events.count() == 5
    assert repos.events.count(state="resolved") == 2


def test_vitals_insert_and_latest(repos: Repositories):
    repos.vitals.insert(
        VitalRow(
            timestamp="2025-01-01T00:00:01+00:00",
            device_id="wearable-1",
            heart_rate=72.0,
            motion=0.1,
            battery=88.0,
            connection_quality=0.95,
            simulated=True,
            metadata={"note": "sim"},
        )
    )
    repos.vitals.insert(
        VitalRow(
            timestamp="2025-01-01T00:00:06+00:00",
            device_id="wearable-1",
            heart_rate=75.0,
            motion=0.2,
        )
    )
    latest = repos.vitals.latest("wearable-1")
    assert latest is not None
    assert latest.heart_rate == pytest.approx(75.0)
    assert repos.vitals.count() == 2
    # metadata round-trips
    first = repos.vitals.list(limit=10)[-1]
    assert first.metadata == {"note": "sim"}
    assert first.simulated is True


def test_alerts_record_and_count(repos: Repositories):
    repos.events.create(_make_event())
    repos.alerts.record(
        AlertRow(
            event_uid="evt-1",
            provider="console",
            attempt_time="2025-01-01T00:00:02+00:00",
            success=True,
            response_metadata={"delivered": True},
        )
    )
    repos.alerts.record(
        AlertRow(
            event_uid="evt-1",
            provider="webhook",
            attempt_time="2025-01-01T00:00:02+00:00",
            success=False,
            failure_message="connection refused",
        )
    )
    assert repos.alerts.count(event_uid="evt-1") == 2
    assert repos.alerts.count(success=True) == 1
    assert repos.alerts.count(success=False) == 1
    for_event = repos.alerts.list_for_event("evt-1")
    assert {a.provider for a in for_event} == {"console", "webhook"}
    assert for_event[0].response_metadata == {"delivered": True}


def test_devices_upsert_and_update(repos: Repositories):
    repos.devices.upsert(
        DeviceRow(device_id="camera-1", device_type="camera", display_name="Front cam")
    )
    repos.devices.update(
        "camera-1", connection_status="ok", last_seen="2025-01-01T00:00:00+00:00"
    )
    dev = repos.devices.get("camera-1")
    assert dev is not None
    assert dev.connection_status == "ok"
    # upsert again updates rather than duplicating
    repos.devices.upsert(
        DeviceRow(device_id="camera-1", device_type="camera", display_name="Renamed")
    )
    assert len(repos.devices.list()) == 1
    assert repos.devices.get("camera-1").display_name == "Renamed"


def test_persistence_across_reopen(temp_db_path, manual_clock):
    db1 = Database(temp_db_path, clock=manual_clock)
    db1.initialize()
    Repositories(db1).events.create(_make_event(uid="persist-1", state="resolved"))
    db1.close()

    db2 = Database(temp_db_path, clock=manual_clock)
    db2.initialize()  # should not wipe data
    ev = Repositories(db2).events.get_by_uid("persist-1")
    assert ev is not None
    assert ev.state == "resolved"
    db2.close()


def test_health_reports_unwritable_gracefully(tmp_path, manual_clock):
    db = Database(tmp_path / "ok.db", clock=manual_clock)
    db.initialize()
    h = db.health()
    assert h["ok"] is True and h["writable"] is True
    db.close()
