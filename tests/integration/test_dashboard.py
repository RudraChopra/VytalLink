"""Integration tests for the dashboard (template + static assets)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vytallink.api.server import create_app
from vytallink.config import load_settings
from vytallink.monitoring import MonitoringService


@pytest.fixture
def client(tmp_path: Path):
    settings = load_settings(
        env="development",
        vision_mode="simulation",
        database_path=str(tmp_path / "dash.db"),
        log_dir=str(tmp_path / "logs"),
        events_dir=str(tmp_path / "events"),
        clips_dir=str(tmp_path / "clips"),
    )
    app = create_app(settings, MonitoringService(settings))
    with TestClient(app) as c:
        yield c


def test_dashboard_html_renders_real_template(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "VytalLink" in body
    assert "Phase 1" in body
    assert "not</strong> a certified medical device" in body  # disclaimer present
    assert "dev-controls" in body  # dev controls element present
    assert "Recent events" in body
    assert "/static/app.js" in body
    assert "/static/style.css" in body
    # No live video feed element.
    assert "<video" not in body.lower()


def test_static_css_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "VytalLink dashboard" in r.text


def test_dashboard_has_patient_panel(client):
    body = client.get("/").text
    # Patient panel elements present for vitals, alert, freshness, vision, incident.
    for el in ("patient-card", "p-alert-level", "p-reasons", "p-hr", "p-rr",
               "p-fresh", "p-vision", "p-source-cam", "p-incident", "p-cameras", "p-snap"):
        assert el in body, f"missing dashboard element: {el}"
    assert "not a medical assessment" in body


def test_dashboard_js_is_xss_safe_and_uses_canonical_endpoint(client):
    js = client.get("/static/app.js").text
    assert "renderPatient" in js
    assert "/api/patient" in js                  # uses the backend's normalized state
    # Reason codes are rendered as textContent (escaped), never injected as HTML.
    assert "c.textContent = reasonLabel(r)" in js
    assert "reasons.innerHTML" not in js


def test_patient_panel_data_available_through_api(client):
    client.post("/api/vitals", json={"heart_rate": 72, "respiratory_rate": 16})
    p = client.get("/api/patient").json()
    # Everything the dashboard renders is present + credential-free.
    assert {"version", "vitals", "vision", "freshness", "alert"} <= set(p)
    assert p["vitals"]["heart_rate"] == 72
    blob = client.get("/api/patient").text
    assert "password" not in blob.lower() and "rtsp://" not in blob
