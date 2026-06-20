"""Stale-incident reconciliation: pure policy + service-level behavior.

Deterministic (ManualClock-free: ages are crafted directly). Resolution uses the
supported resolve path; incidents are never deleted; snapshots are untouched.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from vytallink.common.clock import isoformat
from vytallink.config import load_settings
from vytallink.database.models import EventRow, IncidentVitalRow
from vytallink.monitoring import MonitoringService
from vytallink.monitoring.incident_reconcile import (
    REASON_CAMERA_RECOVERED,
    REASON_STALE_TIMEOUT,
    classify_incident,
)


# --- pure policy -----------------------------------------------------------
def _c(**over):
    base = dict(age_seconds=1000.0, stale_seconds=300.0, source_in_config=True,
                camera_fresh=True, camera_fall_state="normal")
    base.update(over)
    return classify_incident(**base)


def test_recent_incident_is_kept():
    assert _c(age_seconds=100.0) == (False, None, False)


def test_orphaned_camera_resolves():
    assert _c(source_in_config=False) == (True, REASON_STALE_TIMEOUT, False)


def test_ongoing_supported_fall_is_kept():
    for st in ("confirmed_fall", "possible_fall", "recovering"):
        assert _c(camera_fall_state=st) == (False, None, False)


def test_fresh_normal_camera_resolves_as_recovered():
    assert _c(camera_fall_state="normal") == (True, REASON_CAMERA_RECOVERED, False)


def test_configured_offline_camera_is_ambiguous_and_kept():
    assert _c(camera_fresh=False, camera_fall_state=None) == (False, None, True)


# --- service-level ---------------------------------------------------------
def _svc(tmp_path: Path, **over) -> MonitoringService:
    base = dict(env="development", vision_mode="simulation", detector_mode="simulation",
                wearable_mode="simulation", database_path=str(tmp_path / "r.db"),
                log_dir=str(tmp_path / "l"), events_dir=str(tmp_path / "e"), clips_dir=str(tmp_path / "c"),
                disk_warning_percent=100.0, incident_stale_seconds=300.0)
    base.update(over)
    svc = MonitoringService(load_settings(**base))
    svc.db.initialize()
    try:
        svc.camera.open()  # so the simulated camera reads connected+fresh
    except Exception:
        pass
    return svc


def _craft(svc: MonitoringService, uid: str, src: str, *, age_s: float = 4000.0, state: str = "confirmed_fall") -> None:
    old = isoformat(svc.system_clock.now() - timedelta(seconds=age_s))
    svc.repos.events.create(EventRow(event_uid=uid, event_type="fall", state=state,
                                     start_time=old, confirmed_time=old, source_device=src))
    svc.db.execute("UPDATE events SET updated_at=? WHERE event_uid=?", (old, uid))


@pytest.mark.asyncio
async def test_orphaned_stale_incident_resolves_with_reason(tmp_path):
    svc = _svc(tmp_path)  # sim camera id = settings.camera_device_id (e.g. 'camera-1')
    _craft(svc, "evt-orphan", "camera-9")              # camera-9 is not configured -> orphaned
    res = await svc._reconcile_incidents("runtime")
    assert res["resolved"] == 1
    row = svc.repos.events.get_by_uid("evt-orphan")
    assert row.state == "resolved"
    assert REASON_STALE_TIMEOUT in (row.resolution_note or "")
    assert "runtime" in (row.resolution_note or "")


@pytest.mark.asyncio
async def test_recovered_camera_resolves(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-rec", svc.settings.camera_device_id)   # configured + sim camera fresh+normal
    await svc._reconcile_incidents("runtime")
    row = svc.repos.events.get_by_uid("evt-rec")
    assert row.state == "resolved"
    assert REASON_CAMERA_RECOVERED in (row.resolution_note or "")


@pytest.mark.asyncio
async def test_recent_incident_not_resolved(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-recent", "camera-9", age_s=10.0)   # within stale window
    await svc._reconcile_incidents("runtime")
    assert svc.repos.events.get_by_uid("evt-recent").state == "confirmed_fall"


@pytest.mark.asyncio
async def test_configured_offline_camera_kept_and_degrades(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    # Force the source camera to look configured-but-offline (ambiguous).
    monkeypatch.setattr(svc, "_camera_summaries", lambda: [
        {"id": "cam-x", "connected": False, "fall_state": "normal", "frame_age_seconds": None,
         "person_count": None, "fall_confidence": None}])
    monkeypatch.setattr(svc, "_active_camera_ids", lambda: {"cam-x"})
    _craft(svc, "evt-amb", "cam-x")
    res = await svc._reconcile_incidents("runtime")
    assert res["resolved"] == 0 and res["ambiguous"] == 1
    assert svc.repos.events.get_by_uid("evt-amb").state == "confirmed_fall"  # kept open
    assert svc._reconcile_ambiguous_open == 1
    # Health degrades while an ambiguous incident is open.
    svc._running = True
    svc.detector.health = lambda: {"status": "ok", "name": "simulated", "loaded": True}
    assert svc.health()["overall"] == "degraded"


@pytest.mark.asyncio
async def test_reconciliation_is_idempotent(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-orphan", "camera-9")
    r1 = await svc._reconcile_incidents("startup_reconciliation")
    r2 = await svc._reconcile_incidents("runtime")
    assert r1["resolved"] == 1 and r2["resolved"] == 0   # second run is a no-op
    assert svc._incidents_reconciled == 1


@pytest.mark.asyncio
async def test_snapshot_unchanged_after_reconciliation(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-orphan", "camera-9")
    svc.repos.incident_vitals.create(IncidentVitalRow(event_uid="evt-orphan", camera_id="camera-9",
                                                      heart_rate=72, vitals_freshness="fresh"))
    await svc._reconcile_incidents("runtime")
    snap = svc.repos.incident_vitals.get_by_event("evt-orphan")
    assert snap is not None and snap.heart_rate == 72 and snap.vitals_freshness == "fresh"


@pytest.mark.asyncio
async def test_db_failure_is_isolated(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    monkeypatch.setattr(svc.db, "query_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    res = await svc._reconcile_incidents("runtime")   # must NOT raise
    assert res["resolved"] == 0
    assert svc._reconcile_failures == 1


@pytest.mark.asyncio
async def test_resolved_incident_does_not_pin_alert_score(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-orphan", "camera-9")
    # Before: the unresolved incident pins incident_active.
    assert svc._active_incident_id() == "evt-orphan"
    await svc._reconcile_incidents("runtime")
    assert svc._active_incident_id() is None
    assert "incident_active" not in svc.patient_state()["alert"]["reasons"]


@pytest.mark.asyncio
async def test_two_incidents_handled_independently(tmp_path):
    svc = _svc(tmp_path)
    _craft(svc, "evt-orphan", "camera-9")             # orphaned -> resolves
    _craft(svc, "evt-recent", "camera-8", age_s=5.0)  # recent -> kept (independent)
    await svc._reconcile_incidents("runtime")
    assert svc.repos.events.get_by_uid("evt-orphan").state == "resolved"
    assert svc.repos.events.get_by_uid("evt-recent").state == "confirmed_fall"


def test_auto_resolve_disabled_is_a_noop(tmp_path):
    import asyncio
    svc = _svc(tmp_path, incident_auto_resolve_enabled=False)
    _craft(svc, "evt-orphan", "camera-9")
    asyncio.run(svc._reconcile_incidents("runtime"))
    assert svc.repos.events.get_by_uid("evt-orphan").state == "confirmed_fall"  # untouched
