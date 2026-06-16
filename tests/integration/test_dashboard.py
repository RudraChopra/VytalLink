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
