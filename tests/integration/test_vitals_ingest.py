"""Integration tests for iPhone vitals ingestion (POST /api/vitals) and the
enriched /latest, /api/vitals/latest, /api/patient responses.

A real MonitoringService runs in simulation mode against a temp DB. The
simulated wearable interval is set very high so the test's posted iPhone sample
is deterministically the latest vital.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vytallink.api.server import create_app
from vytallink.config import load_settings
from vytallink.monitoring import MonitoringService


@pytest.fixture
def client(tmp_path: Path):
    s = load_settings(
        env="development", vision_mode="simulation", detector_mode="simulation",
        wearable_mode="simulation", database_path=str(tmp_path / "v.db"),
        log_dir=str(tmp_path / "l"), events_dir=str(tmp_path / "e"), clips_dir=str(tmp_path / "c"),
        disk_warning_percent=100.0, wearable_sample_seconds=3600.0,  # don't race the test
    )
    with TestClient(create_app(s, MonitoringService(s))) as c:
        yield c


# --- ingestion validation --------------------------------------------------
def test_valid_payload_accepted_and_reflected(client):
    r = client.post("/api/vitals", json={"heart_rate": 72, "respiratory_rate": 16, "posture": "upright"})
    assert r.status_code == 200
    assert r.json()["accepted"] is True and r.json()["idempotent"] is False
    latest = client.get("/api/vitals/latest").json()
    assert latest["vital"]["heart_rate"] == 72
    assert latest["vital"]["respiratory_rate"] == 16
    assert latest["vital"]["posture"] == "upright"
    assert latest["vital"]["source"] == "iphone"


def test_legacy_field_aliases_accepted(client):
    r = client.post("/api/vitals", json={"hr": 80, "rr": 18, "activity": 0.3})
    assert r.status_code == 200
    assert client.get("/api/vitals/latest").json()["vital"]["heart_rate"] == 80


def test_missing_all_signals_rejected(client):
    assert client.post("/api/vitals", json={"device_id": "iphone-1"}).status_code == 422


def test_invalid_type_rejected(client):
    assert client.post("/api/vitals", json={"heart_rate": "abc"}).status_code == 422


def test_out_of_range_rejected(client):
    assert client.post("/api/vitals", json={"heart_rate": 500}).status_code == 422
    assert client.post("/api/vitals", json={"heart_rate": 5}).status_code == 422


def test_non_finite_rejected(client):
    # 1e400 parses to +inf in Python's json; must be rejected with a 4xx.
    code = client.post("/api/vitals", data="{\"heart_rate\": 1e400}",
                       headers={"content-type": "application/json"}).status_code
    assert code in (400, 422)


def test_future_timestamp_rejected(client):
    r = client.post("/api/vitals", json={"heart_rate": 72, "timestamp": "2999-01-01T00:00:00Z"})
    assert r.status_code == 400
    assert "future" in r.json()["detail"]


def test_too_old_timestamp_rejected(client):
    r = client.post("/api/vitals", json={"heart_rate": 72, "timestamp": "2000-01-01T00:00:00Z"})
    assert r.status_code == 400
    assert "old" in r.json()["detail"]


def test_malformed_json_rejected(client):
    r = client.post("/api/vitals", data="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_oversized_payload_rejected(client):
    big = {"heart_rate": 72, "junk": "x" * 9000}  # > VITALS_INGEST_MAX_BYTES (8192)
    assert client.post("/api/vitals", json=big).status_code == 413


def test_duplicate_sample_is_idempotent(client):
    body = {"heart_rate": 72, "sample_id": "abc-123"}
    assert client.post("/api/vitals", json=body).json()["idempotent"] is False
    assert client.post("/api/vitals", json=body).json()["idempotent"] is True
    # Only one iPhone vital stored for that device.
    items = client.get("/api/vitals", params={"device_id": "iphone-1"}).json()
    assert items["returned"] == 1


def test_only_required_field_accepted(client):
    assert client.post("/api/vitals", json={"heart_rate": 72}).status_code == 200


# --- freshness + scoring through the API -----------------------------------
def test_elevated_hr_changes_alert_reason_without_a_fall(client):
    client.post("/api/vitals", json={"heart_rate": 185})
    ps = client.get("/api/patient").json()
    assert "heart_rate_high" in ps["alert"]["reasons"]
    assert ps["alert"]["level"] in ("warning", "critical")
    assert ps["vision"]["overall_state"] != "confirmed_fall"  # no fake fall created


def test_patient_state_has_freshness_and_is_not_diagnosis(client):
    client.post("/api/vitals", json={"heart_rate": 72})
    ps = client.get("/api/patient").json()
    assert ps["freshness"]["vitals"] in ("fresh", "aging", "stale", "unavailable")
    assert ps["not_a_diagnosis"] is True
    assert "vision" in ps and "alert" in ps


# --- API compatibility + safety -------------------------------------------
def test_latest_and_canonical_share_data_and_keep_legacy_fields(client):
    client.post("/api/vitals", json={"heart_rate": 72})
    a = client.get("/latest").json()
    b = client.get("/api/vitals/latest").json()
    assert a == b
    assert {"vital", "simulated"} <= set(a.keys())                  # legacy
    assert {"vision", "freshness", "alert"} <= set(a.keys())        # new
    blob = client.get("/api/patient").text + client.get("/latest").text
    assert "password" not in blob.lower() and "rtsp://" not in blob and "Traceback" not in blob


# --- compatibility adapter (aliases, conflicts, contract form) -------------
def test_conflicting_aliases_rejected(client):
    # heart_rate and hr present with DIFFERENT values -> 422.
    r = client.post("/api/vitals", json={"heart_rate": 72, "hr": 80})
    assert r.status_code == 422


def test_same_value_aliases_not_a_conflict(client):
    assert client.post("/api/vitals", json={"heart_rate": 72, "hr": 72}).status_code == 200


def test_new_aliases_br_and_timestamp_variants(client):
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    r = client.post("/api/vitals", json={"hr": 70, "br": 14, "recorded_at": now_iso})
    assert r.status_code == 200
    assert client.get("/api/vitals/latest").json()["vital"]["respiratory_rate"] == 14


def test_boolean_as_number_rejected(client):
    assert client.post("/api/vitals", json={"heart_rate": True}).status_code == 422


def test_contract_form_reported(client):
    canon = client.post("/api/vitals", json={"heart_rate": 72}).json()
    alias = client.post("/api/vitals", json={"hr": 72}).json()
    assert canon["contract_form"] == "canonical"
    assert alias["contract_form"] == "alias"
    assert "heart_rate" in canon["accepted_fields"]


def test_patient_state_has_version(client):
    client.post("/api/vitals", json={"heart_rate": 72})
    assert client.get("/api/patient").json()["version"] == 1


# --- failure isolation -----------------------------------------------------
def test_malformed_request_does_not_break_monitoring(client):
    assert client.post("/api/vitals", json={"heart_rate": "bad"}).status_code == 422
    # The app and monitoring loop keep working after a rejected request.
    assert client.get("/health").json()["server"]["running"] is True
    assert client.post("/api/vitals", json={"heart_rate": 72}).status_code == 200
