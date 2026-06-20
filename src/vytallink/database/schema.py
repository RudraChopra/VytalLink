"""SQLite schema definition and migration steps.

Schema versioning uses SQLite's ``PRAGMA user_version``. Migrations are applied
in order from the database's current version up to ``LATEST_SCHEMA_VERSION``.
Phase 1 ships version 1 (the full initial schema). Future schema changes append
new ``(version, statements)`` entries to :data:`MIGRATIONS` — existing data is
preserved across restarts and upgrades.
"""

from __future__ import annotations

LATEST_SCHEMA_VERSION = 2

# --- Version 1: initial schema --------------------------------------------

_V1_STATEMENTS: tuple[str, ...] = (
    # Events: one row per fall (or other) event tracked by the state machine.
    """
    CREATE TABLE IF NOT EXISTS events (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uid         TEXT    NOT NULL UNIQUE,
        event_type        TEXT    NOT NULL DEFAULT 'fall',
        state             TEXT    NOT NULL,
        start_time        TEXT    NOT NULL,
        confirmed_time    TEXT,
        end_time          TEXT,
        resolved_time     TEXT,
        highest_confidence REAL   NOT NULL DEFAULT 0.0,
        detection_count   INTEGER NOT NULL DEFAULT 0,
        source_device     TEXT    NOT NULL DEFAULT 'unknown',
        snapshot_path     TEXT,
        clip_path         TEXT,
        human_label       TEXT,
        resolution_note   TEXT,
        created_at        TEXT    NOT NULL,
        updated_at        TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_state ON events(state)",
    "CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time)",
    "CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_human_label ON events(human_label)",
    # Vitals: wearable readings (simulated in Phase 1).
    """
    CREATE TABLE IF NOT EXISTS vitals (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp          TEXT    NOT NULL,
        device_id          TEXT    NOT NULL,
        heart_rate         REAL,
        motion             REAL,
        connection_quality REAL,
        battery            REAL,
        simulated          INTEGER NOT NULL DEFAULT 1,
        metadata           TEXT,
        created_at         TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_vitals_device ON vitals(device_id)",
    "CREATE INDEX IF NOT EXISTS idx_vitals_timestamp ON vitals(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_vitals_device_ts ON vitals(device_id, timestamp)",
    # Alerts: one row per delivery attempt for an event.
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uid         TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        attempt_time      TEXT    NOT NULL,
        success           INTEGER NOT NULL DEFAULT 0,
        failure_message   TEXT,
        response_metadata TEXT,
        created_at        TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_event ON alerts(event_uid)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_provider ON alerts(provider)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_attempt ON alerts(attempt_time)",
    # Devices: known devices and their connection state.
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id         TEXT    PRIMARY KEY,
        device_type       TEXT    NOT NULL,
        display_name      TEXT    NOT NULL DEFAULT '',
        connection_status TEXT    NOT NULL DEFAULT 'unknown',
        last_seen         TEXT,
        last_error        TEXT,
        metadata          TEXT,
        created_at        TEXT    NOT NULL,
        updated_at        TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_devices_type ON devices(device_type)",
)

# v2: one vitals snapshot per confirmed fall incident (additive, backward-compatible).
# UNIQUE(event_uid) enforces exactly one snapshot per logical incident at the DB
# layer. No credentials, RTSP URLs, or raw payloads are ever stored here.
_V2_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS incident_vitals (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uid            TEXT    NOT NULL UNIQUE,
        camera_id            TEXT    NOT NULL,
        confirmed_time       TEXT,
        vitals_sample_id     TEXT,
        heart_rate           REAL,
        respiratory_rate     REAL,
        posture              TEXT,
        phone_alert_score    REAL,
        computed_alert_level TEXT,
        computed_alert_score INTEGER,
        reason_codes         TEXT,
        source_timestamp     TEXT,
        received_at          TEXT,
        vitals_age_seconds   REAL,
        vitals_freshness     TEXT,
        vitals_available     INTEGER NOT NULL DEFAULT 0,
        vitals_source        TEXT,
        synthetic            INTEGER NOT NULL DEFAULT 0,
        snapshot_version     INTEGER NOT NULL DEFAULT 1,
        created_at           TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_incident_vitals_camera ON incident_vitals(camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_incident_vitals_created ON incident_vitals(created_at)",
)

# Ordered migrations: (target_version, tuple_of_sql_statements).
MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, _V1_STATEMENTS),
    (2, _V2_STATEMENTS),
)
