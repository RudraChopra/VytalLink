"""Tests for the self-diagnostics module."""

from __future__ import annotations

from vytallink.config import load_settings
from vytallink.diagnostics import FAIL, overall_status, run_diagnostics


def test_diagnostics_no_failures_in_simulation(tmp_path):
    settings = load_settings(
        env="development",
        vision_mode="simulation",
        detector_mode="simulation",
        database_path=str(tmp_path / "diag.db"),
        log_dir=str(tmp_path / "logs"),
    )
    checks = run_diagnostics(settings)
    names = {c.name for c in checks}
    assert {
        "environment", "python_env", "imports", "configuration", "database",
        "port", "gpu", "camera_config", "model_config", "wearable", "disk",
    } <= names
    # No hard failures expected for a default simulation configuration.
    failures = [c for c in checks if c.status == FAIL]
    assert failures == [], f"unexpected failures: {[(c.name, c.detail) for c in failures]}"
    assert overall_status(checks) in ("PASS", "WARN")


def test_database_check_reports_status(tmp_path):
    settings = load_settings(
        database_path=str(tmp_path / "diag2.db"), log_dir=str(tmp_path / "logs")
    )
    checks = {c.name: c for c in run_diagnostics(settings)}
    assert checks["database"].status == "PASS"
    assert "schema v" in checks["database"].detail
