"""Incident vitals snapshot tests: one snapshot per confirmed incident, correct
camera + vitals + freshness + reason codes, dedup, synthetic marking, failure
isolation. Simulation mode + ManualClock — deterministic, no hardware, no sleeps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vytallink.api.server import create_app
from vytallink.config import load_settings
from vytallink.events.manager import EventManager
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.monitoring import MonitoringService


def _make_client(tmp_path: Path, **over):
    base = dict(
        env="development", vision_mode="simulation", detector_mode="simulation",
        wearable_mode="simulation", database_path=str(tmp_path / "snap.db"),
        log_dir=str(tmp_path / "l"), events_dir=str(tmp_path / "e"), clips_dir=str(tmp_path / "c"),
        disk_warning_percent=100.0, wearable_sample_seconds=3600.0,
        fall_confirm_seconds=2.0, fall_clear_seconds=3.0,
    )
    base.update(over)
    s = load_settings(**base)
    svc = MonitoringService(s)
    c = TestClient(create_app(s, svc))
    c._svc = svc  # type: ignore[attr-defined]
    return c


@pytest.fixture
def client(tmp_path):
    c = _make_client(tmp_path)
    with c:
        yield c


def test_confirmed_fall_creates_one_snapshot_with_vitals(client):
    svc = client._svc
    client.post("/api/vitals", json={"heart_rate": 72, "respiratory_rate": 16, "posture": "upright"})
    client.post("/api/simulation/fall")
    snaps = svc.repos.incident_vitals.list()
    assert len(snaps) == 1
    s = snaps[0]
    assert s.camera_id == svc.settings.camera_device_id
    assert s.heart_rate == 72 and s.respiratory_rate == 16 and s.posture == "upright"
    assert s.vitals_freshness == "fresh" and s.vitals_available is True
    assert s.computed_alert_level == "critical"
    assert "fall_confirmed" in s.reason_codes
    assert s.synthetic is False


def test_ongoing_confirmed_fall_does_not_duplicate_snapshot(client):
    svc = client._svc
    client.post("/api/vitals", json={"heart_rate": 72})
    client.post("/api/simulation/fall")
    # Re-trigger while still confirmed: no new incident -> no new snapshot.
    client.post("/api/simulation/fall")
    assert svc.repos.incident_vitals.count() == 1


def test_resolved_then_new_fall_creates_second_snapshot(client):
    svc = client._svc
    client.post("/api/vitals", json={"heart_rate": 72})
    client.post("/api/simulation/fall")
    client.post("/api/simulation/normal")   # resolve
    client.post("/api/vitals", json={"heart_rate": 99})
    client.post("/api/simulation/fall")     # independent new incident
    snaps = svc.repos.incident_vitals.list()
    assert len(snaps) == 2
    assert {s.event_uid for s in snaps} == set(s.event_uid for s in snaps)  # distinct uids
    assert len({s.event_uid for s in snaps}) == 2


def test_snapshot_exposed_on_event_detail_api(client):
    client.post("/api/vitals", json={"heart_rate": 72})
    client.post("/api/simulation/fall")
    uid = client.get("/api/events").json()["items"][0]["event_uid"]
    detail = client.get(f"/api/events/{uid}").json()
    assert detail["incident_vitals"] is not None
    assert detail["incident_vitals"]["camera_id"] == client._svc.settings.camera_device_id
    assert "heart_rate" in detail["incident_vitals"]
    # No credentials/internals in the detail payload.
    assert "password" not in client.get(f"/api/events/{uid}").text.lower()


def test_synthetic_incident_creates_synthetic_snapshot(tmp_path):
    # 'standing' triggers synthetic mode; the rest keep the simulated fall working.
    c = _make_client(tmp_path, fall_class_names="standing,fall,fallen,lying,fall_detected,person_fall",
                     allow_synthetic_fall_testing=True)
    with c:
        assert c._svc.synthetic_mode is True
        c.post("/api/vitals", json={"heart_rate": 72})
        c.post("/api/simulation/fall")
        snaps = c._svc.repos.incident_vitals.list()
        assert len(snaps) == 1
        assert snaps[0].synthetic is True


def test_two_cameras_produce_independent_snapshots(client):
    # Per-camera attribution: each confirming camera's source_device flows to its
    # own snapshot, and two coexist (the core multi-camera incident behavior).
    from types import SimpleNamespace

    svc = client._svc
    svc._write_incident_snapshot(SimpleNamespace(event_uid="evt-c1", source_device="camera_1", confirmed_time=None))
    svc._write_incident_snapshot(SimpleNamespace(event_uid="evt-c2", source_device="camera_2", confirmed_time=None))
    s1 = svc.repos.incident_vitals.get_by_event("evt-c1")
    s2 = svc.repos.incident_vitals.get_by_event("evt-c2")
    assert s1 is not None and s2 is not None
    assert s1.camera_id == "camera_1" and s2.camera_id == "camera_2"
    assert {s.event_uid for s in svc.repos.incident_vitals.list()} >= {"evt-c1", "evt-c2"}
    # A duplicate write for camera_1's incident does not create a second row.
    svc._write_incident_snapshot(SimpleNamespace(event_uid="evt-c1", source_device="camera_1", confirmed_time=None))
    assert sum(1 for s in svc.repos.incident_vitals.list() if s.event_uid == "evt-c1") == 1


def test_health_reports_snapshot_writer(client):
    client.post("/api/vitals", json={"heart_rate": 72})
    client.post("/api/simulation/fall")
    p = client.get("/health").json()["persistence"]
    assert p["snapshot_writer"] == "ok"
    assert p["snapshots_written"] >= 1
    assert p["snapshot_failures"] == 0
    assert p["incident_snapshots_total"] >= 1


# --- failure isolation (EventManager level) --------------------------------
@pytest.mark.asyncio
async def test_snapshot_failure_does_not_break_persistence(repos, manual_clock):
    def boom(ev):  # noqa: ANN001
        raise RuntimeError("snapshot DB down")

    sm = FallEventStateMachine(confirm_seconds=0.5, clear_seconds=1.0, cooldown_seconds=0.0,
                               source_device="camera_1", clock=manual_clock)
    em = EventManager(repos, sm, None, clock=manual_clock, simulated=False, snapshot_fn=boom)
    await em.observe(True, 0.9)
    manual_clock.advance(0.6)
    await em.observe(True, 0.9)   # must NOT raise though the snapshot hook throws
    assert repos.events.count() == 1   # event still persisted
