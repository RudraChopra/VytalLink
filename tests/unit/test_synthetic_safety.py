"""Synthetic fall-testing safety: fail-closed config, event marking, alert
dry-run, and the synthetic-only cleanup tool.

Synthetic mode treats a non-fall posture (e.g. 'standing') as fall evidence so
the live pipeline can be validated without staging a real fall. It is unsafe to
leave on, so it fails closed (never in production; dev requires an explicit
allow), forces alerts to dry-run, and tags events event_type='fall_synthetic'.
"""

from __future__ import annotations

import pytest

from vytallink.common.errors import ConfigError
from vytallink.config import load_settings
from vytallink.database.maintenance import (
    count_real_events,
    delete_synthetic_events,
    list_synthetic_events,
)
from vytallink.database.models import EventRow
from vytallink.events.manager import EventManager
from vytallink.events.state_machine import FallEventStateMachine


def _sim(**over):
    base = dict(env="development", vision_mode="simulation", detector_mode="simulation",
                wearable_mode="simulation", disk_warning_percent=100.0)
    base.update(over)
    return load_settings(**base)


# --- config fail-closed ----------------------------------------------------
def test_default_config_is_not_synthetic():
    s = _sim()
    assert s.synthetic_detection_active is False
    assert s.synthetic_override_in_fall_classes is False


def test_posture_in_fall_classes_requires_explicit_allow():
    # 'standing' as fall evidence, dev, not allowed -> fail closed.
    with pytest.raises(ConfigError):
        _sim(fall_class_names="standing,fallen")


def test_explicit_flag_requires_allow():
    with pytest.raises(ConfigError):
        _sim(synthetic_fall_test_mode=True)


def test_posture_override_allowed_in_dev_with_flag():
    s = _sim(fall_class_names="standing,fallen", allow_synthetic_fall_testing=True)
    assert s.synthetic_detection_active is True
    assert s.synthetic_override_in_fall_classes is True


def test_synthetic_never_allowed_in_production():
    # Even with the allow flag, production must fail closed.
    with pytest.raises(ConfigError):
        _sim(env="production", fall_class_names="standing,fallen", allow_synthetic_fall_testing=True)
    with pytest.raises(ConfigError):
        _sim(env="production", synthetic_fall_test_mode=True, allow_synthetic_fall_testing=True)


# --- alert dry-run ---------------------------------------------------------
def test_dispatcher_dry_run_suppresses_external_delivery(repos):
    from vytallink.alerts.factory import build_dispatcher

    s = _sim(webhook_url="https://example.invalid/hook",
             allow_synthetic_fall_testing=True, synthetic_fall_test_mode=True)
    dry = build_dispatcher(s, repos, dry_run=True)
    full = build_dispatcher(s, repos, dry_run=False)
    assert len(dry.providers) == 1   # console only — external webhook suppressed
    assert len(full.providers) == 2  # console + webhook


# --- persistence marking ---------------------------------------------------
@pytest.mark.asyncio
async def test_synthetic_event_is_tagged_in_persistence(repos, manual_clock):
    sm = FallEventStateMachine(confirm_seconds=0.5, clear_seconds=1.0, cooldown_seconds=0.0,
                               source_device="camera_1", clock=manual_clock)
    em = EventManager(repos, sm, None, clock=manual_clock, simulated=False, synthetic=True)
    await em.observe(True, 0.9)
    manual_clock.advance(0.6)
    await em.observe(True, 0.9)
    ev = repos.events.list()[0]
    assert ev.event_type == "fall_synthetic"


# --- cleanup tool (synthetic-only, refuses real events) --------------------
def test_cleanup_deletes_only_synthetic_events(repos, database):
    repos.events.create(EventRow(event_uid="real-1", event_type="fall",
                                 state="confirmed_fall", start_time="2026-01-01T00:00:00Z",
                                 source_device="camera_1"))
    repos.events.create(EventRow(event_uid="syn-1", event_type="fall_synthetic",
                                 state="confirmed_fall", start_time="2026-01-01T00:00:00Z",
                                 source_device="camera_2"))
    assert len(list_synthetic_events(database)) == 1
    assert count_real_events(database) == 1

    n = delete_synthetic_events(database)
    assert n == 1
    assert list_synthetic_events(database) == []
    assert count_real_events(database) == 1                  # real event preserved
    assert repos.events.get_by_uid("real-1") is not None     # real event still present
    assert repos.events.get_by_uid("syn-1") is None          # synthetic removed


def test_cleanup_is_noop_when_no_synthetic(repos, database):
    repos.events.create(EventRow(event_uid="real-1", event_type="fall",
                                 state="confirmed_fall", start_time="2026-01-01T00:00:00Z",
                                 source_device="camera_1"))
    assert delete_synthetic_events(database) == 0
    assert repos.events.get_by_uid("real-1") is not None


# --- health flag (synthetic mode visible, credential-free) -----------------
def test_health_reports_synthetic_mode(tmp_path):
    from fastapi.testclient import TestClient
    from vytallink.api.server import create_app
    from vytallink.monitoring import MonitoringService

    s = load_settings(env="development", vision_mode="simulation", detector_mode="simulation",
                      wearable_mode="simulation", database_path=str(tmp_path / "syn.db"),
                      log_dir=str(tmp_path / "l"), events_dir=str(tmp_path / "e"),
                      clips_dir=str(tmp_path / "c"), disk_warning_percent=100.0,
                      fall_class_names="standing,fallen", allow_synthetic_fall_testing=True)
    with TestClient(create_app(s, MonitoringService(s))) as c:
        h = c.get("/health").json()
        assert h["synthetic_detection_mode"] is True
        assert "password" not in c.get("/health").text.lower()
