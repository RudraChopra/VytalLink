"""Schema migration tests: fresh DB, upgrade from v1, idempotent re-run, and
existing event data preserved across the incident_vitals (v2) migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vytallink.database.db import Database
from vytallink.database.models import EventRow, IncidentVitalRow
from vytallink.database.repositories import Repositories
from vytallink.database.schema import LATEST_SCHEMA_VERSION, _V1_STATEMENTS


def test_fresh_database_migrates_to_latest(tmp_path: Path):
    db = Database(str(tmp_path / "fresh.db"))
    assert db.initialize() == LATEST_SCHEMA_VERSION
    # incident_vitals table exists and is usable.
    repos = Repositories(db)
    assert repos.incident_vitals.count() == 0
    db.close()


def test_repeated_migration_is_idempotent(tmp_path: Path):
    p = str(tmp_path / "idem.db")
    Database(p).initialize()
    db = Database(p)
    assert db.initialize() == LATEST_SCHEMA_VERSION   # second run: no-op, no error
    db.close()


def test_upgrade_from_v1_preserves_events(tmp_path: Path):
    p = str(tmp_path / "v1.db")
    # Build a v1 database by hand (only the v1 statements + user_version=1).
    conn = sqlite3.connect(p)
    for stmt in _V1_STATEMENTS:
        conn.execute(stmt)
    conn.execute(
        "INSERT INTO events (event_uid, event_type, state, start_time, source_device, created_at, updated_at) "
        "VALUES ('evt-old','fall','confirmed_fall','t','camera_1','t','t')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit(); conn.close()

    db = Database(p)
    assert db.initialize() == LATEST_SCHEMA_VERSION   # upgrades v1 -> latest
    repos = Repositories(db)
    assert repos.events.get_by_uid("evt-old") is not None   # existing event preserved
    # New table works on the upgraded DB.
    repos.incident_vitals.create(IncidentVitalRow(event_uid="evt-old", camera_id="camera_1"))
    assert repos.incident_vitals.count() == 1
    db.close()


def test_snapshot_unique_per_incident(tmp_path: Path):
    db = Database(str(tmp_path / "uniq.db"))
    db.initialize()
    repos = Repositories(db)
    repos.events.create(EventRow(event_uid="evt-1", state="confirmed_fall",
                                 start_time="t", source_device="camera_1"))
    _, c1 = repos.incident_vitals.create(IncidentVitalRow(event_uid="evt-1", camera_id="camera_1", heart_rate=72))
    _, c2 = repos.incident_vitals.create(IncidentVitalRow(event_uid="evt-1", camera_id="camera_1", heart_rate=80))
    assert c1 is True and c2 is False          # second insert ignored (one per incident)
    assert repos.incident_vitals.count() == 1
    assert repos.incident_vitals.get_by_event("evt-1").heart_rate == 72   # original kept
    db.close()
