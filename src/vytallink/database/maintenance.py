"""Maintenance helpers for synthetic (validation) fall-event cleanup.

Synthetic events — created while synthetic fall testing was active — are tagged
``event_type='fall_synthetic'`` (an explicit marker, NOT identified by low
confidence). Every helper here matches ONLY that exact marker, so a real event
(``event_type='fall'``) can never be selected or deleted. Parameterized SQL only.
"""

from __future__ import annotations

from typing import Any

from vytallink.database.db import Database
from vytallink.events.manager import EventManager

SYNTHETIC_EVENT_TYPE = EventManager.SYNTHETIC_EVENT_TYPE
_SAFE_COLUMNS = (
    "event_uid, event_type, state, start_time, source_device, "
    "highest_confidence, human_label, created_at"
)


def list_synthetic_events(db: Database) -> list[dict[str, Any]]:
    """All events explicitly tagged synthetic, with safe (no-path) columns."""
    rows = db.query_all(
        f"SELECT {_SAFE_COLUMNS} FROM events WHERE event_type = ? ORDER BY created_at",
        (SYNTHETIC_EVENT_TYPE,),
    )
    return [dict(r) for r in rows]


def count_real_events(db: Database) -> int:
    """Count of non-synthetic events (the ones cleanup must never touch)."""
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM events WHERE event_type != ?", (SYNTHETIC_EVENT_TYPE,)
    )
    return int(row["n"]) if row else 0


def delete_synthetic_events(db: Database) -> int:
    """Delete ONLY rows tagged synthetic (and their alert rows, to avoid orphans).
    Real events are never matched. Returns the number of events deleted."""
    uids = [
        r["event_uid"]
        for r in db.query_all(
            "SELECT event_uid FROM events WHERE event_type = ?", (SYNTHETIC_EVENT_TYPE,)
        )
    ]
    for uid in uids:
        db.execute("DELETE FROM alerts WHERE event_uid = ?", (uid,))
    db.execute("DELETE FROM events WHERE event_type = ?", (SYNTHETIC_EVENT_TYPE,))
    return len(uids)
