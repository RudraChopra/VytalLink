"""Tests for configuration defaults, validation, and secret sanitization."""

from __future__ import annotations

import pytest

from vytallink.common.errors import ConfigError
from vytallink.common.sanitize import mask_value, sanitize_secret, sanitize_url
from vytallink.config import DetectorMode, Environment, VisionMode, load_settings


def test_defaults_enable_simulation():
    s = load_settings()
    assert s.env == Environment.DEVELOPMENT
    assert s.port == 5050
    assert s.vision_mode == VisionMode.SIMULATION
    assert s.detector_mode == DetectorMode.SIMULATION
    assert s.simulation_active is True
    assert s.confidence_threshold == pytest.approx(0.55)


def test_missing_optional_hardware_ok_in_dev():
    # No camera source / model path, but development simulation must still load.
    s = load_settings(env="development", vision_mode="simulation")
    assert s.camera_source == ""
    assert s.model_path == ""
    assert s.simulation_active is True


def test_production_rtsp_requires_camera_source():
    with pytest.raises(ConfigError) as exc:
        load_settings(env="production", vision_mode="rtsp")
    assert "CAMERA_SOURCE" in str(exc.value)


def test_production_yolo_requires_existing_model(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_settings(env="production", detector_mode="yolo", model_path="/no/such/model.pt")
    assert "MODEL_PATH" in str(exc.value)
    # An existing path passes that particular check.
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    s = load_settings(
        env="production",
        detector_mode="yolo",
        model_path=str(model),
        vision_mode="simulation",
    )
    assert s.model_path == str(model)


def test_invalid_values_rejected():
    with pytest.raises(Exception):
        load_settings(confidence_threshold=1.5)
    with pytest.raises(Exception):
        load_settings(port=70000)
    with pytest.raises(Exception):
        load_settings(log_level="LOUD")
    with pytest.raises(Exception):
        load_settings(process_every_n_frames=0)


def test_fall_class_set_parsing():
    s = load_settings(fall_class_names="Fall, fallen ,LYING")
    assert s.fall_class_set == {"fall", "fallen", "lying"}


def test_sanitize_url_redacts_credentials():
    out = sanitize_url("rtsp://alice:s3cret@cam.local:554/Streaming/Channels/101")
    assert "alice" not in out
    assert "s3cret" not in out
    assert "cam.local:554" in out
    assert out.startswith("rtsp://***REDACTED***@")


def test_sanitize_url_without_credentials_unchanged():
    url = "rtsp://cam.local:554/stream"
    assert sanitize_url(url) == url
    assert sanitize_url("") == ""


def test_sanitize_url_password_containing_at_sign():
    # A password containing '@' must not leak any part of itself.
    out = sanitize_url("rtsp://alice:p@ss@cam.local:554/stream")
    assert "p@ss" not in out
    assert "ss@" not in out
    assert "alice" not in out
    assert "cam.local:554" in out
    assert out == "rtsp://***REDACTED***@cam.local:554/stream"


def test_sanitize_url_scheme_less_credentials():
    out = sanitize_url("user:secretpw@host:554/path")
    assert "secretpw" not in out
    assert "user" not in out
    assert "host:554" in out
    assert out.startswith("***REDACTED***@")


def test_sanitize_url_ipv6_host():
    out = sanitize_url("rtsp://bob:pw@[2001:db8::1]:554/s")
    assert "pw" not in out and "bob" not in out
    assert "[2001:db8::1]:554" in out


def test_sanitize_secret_and_mask():
    assert sanitize_secret("hunter2") == "***REDACTED***"
    assert sanitize_secret("") == ""
    assert sanitize_secret(None) == ""
    assert mask_value("abcdef", keep=2) == "…ef"
    assert mask_value("abcdef") == "***REDACTED***"


def test_safe_summary_never_leaks_secrets():
    s = load_settings(
        vision_mode="rtsp",
        camera_source="rtsp://cam.local:554/s",
        camera_username="bob",
        camera_password="topsecret",
        webhook_url="https://user:pw@hooks.example.com/x",
        webhook_secret="whsec_123",
    )
    summary = s.safe_summary()
    blob = str(summary)
    assert "topsecret" not in blob
    assert "whsec_123" not in blob
    assert "pw@hooks" not in blob
    assert summary["camera_password"] == "***REDACTED***"
    assert summary["webhook_secret"] == "***REDACTED***"


def test_camera_connection_embeds_credentials_but_summary_hides():
    s = load_settings(
        vision_mode="rtsp",
        camera_source="rtsp://cam.local:554/s",
        camera_username="bob",
        camera_password="pw",
    )
    # The internal connection string is usable...
    assert s.camera_connection_string() == "rtsp://bob:pw@cam.local:554/s"
    # ...but the sanitized form (used for logs/health) is redacted.
    assert "bob" not in s.sanitized_camera_source()
    assert "pw" not in s.sanitized_camera_source()


# --- one Tapo RTSP camera: component-based config (CAMERA_HOST/PORT/PATH) ---
def test_tapo_rtsp_url_assembled_from_components():
    """A Tapo camera is configured from discrete fields (no full CAMERA_SOURCE).
    The assembled URL must embed creds and the stream path correctly."""
    s = load_settings(
        vision_mode="rtsp",
        camera_host="192.168.1.71",
        camera_port=554,
        camera_stream_path="stream1",  # leading slash optional
        camera_username="vyt",
        camera_password="secretpw",
    )
    assert s.camera_connection_string() == "rtsp://vyt:secretpw@192.168.1.71:554/stream1"
    assert s.has_camera_target is True


def test_tapo_rtsp_password_special_chars_encoded_and_redacted():
    """A password with RTSP-significant characters must be URL-encoded in the
    connection string yet fully redacted everywhere it is surfaced."""
    s = load_settings(
        vision_mode="rtsp",
        camera_host="192.168.1.71",
        camera_stream_path="/stream1",
        camera_username="vyt",
        camera_password="p@ss:w/d",
    )
    conn = s.camera_connection_string()
    # Encoded so the URL stays well-formed (@ -> %40, : -> %3A, / -> %2F).
    assert "p%40ss%3Aw%2Fd" in conn
    assert "p@ss:w/d" not in conn
    # Redacted in every public/log surface.
    redacted = s.sanitized_camera_source()
    assert "p@ss" not in redacted and "p%40ss" not in redacted
    assert "vyt" not in redacted
    assert redacted == "rtsp://***REDACTED***@192.168.1.71:554/stream1"
    blob = str(s.safe_summary())
    assert "p@ss" not in blob and "p%40ss" not in blob
    assert s.safe_summary()["camera_password"] == "***REDACTED***"


def test_tapo_rtsp_camera_redacts_credentials_in_provider():
    """The actual camera provider (reused pipeline) must never expose creds in
    its safe_source or health."""
    from vytallink.vision.rtsp import RTSPCamera

    s = load_settings(
        vision_mode="rtsp",
        camera_host="192.168.1.71",
        camera_stream_path="/stream1",
        camera_username="vyt",
        camera_password="secretpw",
    )
    cam = RTSPCamera(s.camera_connection_string(), source_id=s.camera_device_id)
    assert "secretpw" not in cam.safe_source and "vyt" not in cam.safe_source
    assert "192.168.1.71:554" in cam.safe_source
    assert "secretpw" not in str(cam.health())


def test_committed_defaults_keep_rtsp_disabled():
    """Regression for the live-test work: the default (committed) config must
    stay in simulation so RTSP is never enabled by accident."""
    s = load_settings()
    assert s.vision_mode == VisionMode.SIMULATION
    assert s.detector_mode == DetectorMode.SIMULATION
    assert s.has_camera_target is False
