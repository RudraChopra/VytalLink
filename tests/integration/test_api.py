"""Integration tests for the FastAPI app via TestClient.

A real MonitoringService runs in simulation mode against a temp DB. The
TestClient context manager runs the lifespan, so background loops start/stop
and the simulation controls drive the *real* pipeline deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vytallink.api.server import create_app
from vytallink.config import load_settings
from vytallink.monitoring import MonitoringService


def _settings(tmp_path: Path, **over):
    base = dict(
        env="development",
        vision_mode="simulation",
        detector_mode="simulation",
        wearable_mode="simulation",
        database_path=str(tmp_path / "api_test.db"),
        log_dir=str(tmp_path / "logs"),
        events_dir=str(tmp_path / "events"),
        clips_dir=str(tmp_path / "clips"),
        wearable_sample_seconds=0.5,
        fall_confirm_seconds=2.0,
        fall_clear_seconds=3.0,
    )
    base.update(over)
    return load_settings(**base)


@pytest.fixture
def client(tmp_path):
    settings = _settings(tmp_path)
    service = MonitoringService(settings)
    app = create_app(settings, service)
    with TestClient(app) as c:
        c._service = service  # type: ignore[attr-defined]
        yield c


def test_health_has_all_required_fields(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    for key in [
        "overall", "server", "database", "camera", "detector", "wearable",
        "alerts", "gpu", "latest_frame_time", "latest_inference_time",
        "fall_state", "uptime_seconds", "disk", "disk_warning", "simulation",
    ]:
        assert key in h, f"missing health key: {key}"
    assert h["server"]["running"] is True
    assert h["database"]["status"] == "ok"
    assert "password" not in r.text.lower()


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    s = r.json()
    assert s["name"] == "VytalLink"
    assert s["simulation_active"] is True
    assert s["controls_enabled"] is True
    assert s["latest_vital"] is not None  # primed at startup
    assert s["latest_vital"]["simulated"] is True


def test_events_empty_initially(client):
    r = client.get("/api/events")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_simulated_fall_creates_one_event_and_one_alert(client):
    assert client.post("/api/simulation/fall").status_code == 200
    events = client.get("/api/events").json()
    assert events["total"] == 1
    ev = events["items"][0]
    assert ev["state"] == "confirmed_fall"
    uid = ev["event_uid"]
    detail = client.get(f"/api/events/{uid}").json()
    assert detail["alert_count"] == 1  # console only
    assert detail["alert_delivered"] is True
    status = client.get("/api/status").json()
    assert status["counts"]["events"] == 1
    assert status["counts"]["alerts"] == 1
    assert status["fall_state"] == "confirmed_fall"


def test_duplicate_fall_suppressed(client):
    client.post("/api/simulation/fall")
    client.post("/api/simulation/fall")  # repeated evidence
    client.post("/api/simulation/fall")
    events = client.get("/api/events").json()
    assert events["total"] == 1
    status = client.get("/api/status").json()
    assert status["counts"]["alerts"] == 1


def test_label_and_resolve_flow(client):
    client.post("/api/simulation/fall")
    uid = client.get("/api/events").json()["items"][0]["event_uid"]

    # Label as real fall.
    r = client.post(f"/api/events/{uid}/label", json={"label": "real_fall"})
    assert r.status_code == 200
    assert r.json()["human_label"] == "real_fall"

    # Resolve with a note.
    r = client.post(f"/api/events/{uid}/resolve", json={"note": "caregiver checked"})
    assert r.status_code == 200
    assert r.json()["state"] == "resolved"
    assert r.json()["resolution_note"] == "caregiver checked"


def test_invalid_label_rejected(client):
    client.post("/api/simulation/fall")
    uid = client.get("/api/events").json()["items"][0]["event_uid"]
    r = client.post(f"/api/events/{uid}/label", json={"label": "definitely_not_valid"})
    assert r.status_code == 422  # pydantic validation


def test_missing_event_returns_404(client):
    r = client.get("/api/events/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_invalid_query_param_rejected(client):
    r = client.get("/api/events?limit=0")
    assert r.status_code == 422
    r = client.get("/api/events?limit=99999")
    assert r.status_code == 422


def test_devices_listed(client):
    r = client.get("/api/devices")
    assert r.status_code == 200
    types = {d["device_type"] for d in r.json()["items"]}
    assert {"camera", "wearable"} <= types


def test_vitals_endpoints(client):
    latest = client.get("/api/vitals/latest").json()
    assert latest["vital"] is not None
    assert latest["simulated"] is True
    listing = client.get("/api/vitals?limit=10").json()
    assert listing["returned"] >= 1
    assert listing["simulated"] is True


def test_simulation_reset(client):
    client.post("/api/simulation/fall")
    assert client.get("/api/status").json()["fall_state"] == "confirmed_fall"
    assert client.post("/api/simulation/reset").status_code == 200
    assert client.get("/api/status").json()["fall_state"] == "normal"


def test_second_event_after_resolve_and_cooldown(client):
    # First event + alert.
    client.post("/api/simulation/fall")
    client.post("/api/simulation/normal")  # resolve it
    # Second event. The simulation advances the event clock by only a few
    # seconds, which is within the default 30s cooldown, so the event is
    # created but its alert is suppressed.
    client.post("/api/simulation/fall")
    events = client.get("/api/events").json()
    assert events["total"] == 2
    alerts = client.get("/api/status").json()["counts"]["alerts"]
    assert alerts == 1  # second alert suppressed by cooldown


def test_dashboard_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_simulation_disabled_outside_development(tmp_path):
    settings = _settings(tmp_path, env="production")
    service = MonitoringService(settings)
    app = create_app(settings, service)
    with TestClient(app) as c:
        r = c.post("/api/simulation/fall")
        assert r.status_code == 403
        assert r.json()["error"] == "forbidden"
