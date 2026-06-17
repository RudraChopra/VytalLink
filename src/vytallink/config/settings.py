"""Application settings, loaded from environment / ``.env`` and validated.

Design goals (from the Phase 1 spec):

* Missing *optional* hardware values must not prevent simulation mode.
* Missing *required* production values must produce clear startup errors.
* Secrets must never appear in logs — :meth:`Settings.safe_summary` redacts.
* URLs containing passwords are sanitized before logging.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vytallink.common.errors import ConfigError
from vytallink.common.sanitize import sanitize_secret, sanitize_url

# Repo root: .../src/vytallink/config/settings.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class VisionMode(str, Enum):
    SIMULATION = "simulation"
    FILE = "file"
    RTSP = "rtsp"


class DetectorMode(str, Enum):
    SIMULATION = "simulation"
    YOLO = "yolo"
    TENSORRT = "tensorrt"


class WearableMode(str, Enum):
    SIMULATION = "simulation"
    # Future: "ble", "vendor_api", etc.


class Settings(BaseSettings):
    """Validated configuration. Field env names mirror ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ---- Core service -----------------------------------------------------
    env: Environment = Field(default=Environment.DEVELOPMENT, validation_alias="VYTALLINK_ENV")
    host: str = Field(default="0.0.0.0", validation_alias="VYTALLINK_HOST")
    port: int = Field(default=5050, validation_alias="VYTALLINK_PORT")
    database_path: Path = Field(
        default=PROJECT_ROOT / "data" / "database" / "vytallink.db",
        validation_alias="VYTALLINK_DATABASE_PATH",
    )
    log_level: str = Field(default="INFO", validation_alias="VYTALLINK_LOG_LEVEL")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", validation_alias="VYTALLINK_LOG_DIR")

    # ---- Vision / camera --------------------------------------------------
    vision_mode: VisionMode = Field(default=VisionMode.SIMULATION, validation_alias="VISION_MODE")
    # An RTSP URL or video-file path can be given directly via CAMERA_SOURCE, OR
    # the RTSP URL can be assembled from the component fields below. CAMERA_SOURCE
    # (when set) takes precedence. Credentials are kept separate so they are never
    # part of a logged/echoed source string.
    camera_source: str = Field(default="", validation_alias="CAMERA_SOURCE")
    camera_host: str = Field(default="", validation_alias="CAMERA_HOST")
    camera_port: int = Field(default=554, validation_alias="CAMERA_PORT")
    camera_stream_path: str = Field(default="", validation_alias="CAMERA_STREAM_PATH")
    camera_username: str = Field(default="", validation_alias="CAMERA_USERNAME")
    camera_password: str = Field(default="", validation_alias="CAMERA_PASSWORD")
    camera_device_id: str = Field(default="camera-1", validation_alias="CAMERA_DEVICE_ID")

    # ---- Detector / model -------------------------------------------------
    detector_mode: DetectorMode = Field(
        default=DetectorMode.SIMULATION, validation_alias="DETECTOR_MODE"
    )
    model_path: str = Field(default="", validation_alias="MODEL_PATH")
    confidence_threshold: float = Field(default=0.55, validation_alias="CONFIDENCE_THRESHOLD")
    process_every_n_frames: int = Field(default=3, validation_alias="PROCESS_EVERY_N_FRAMES")
    image_size: int = Field(default=416, validation_alias="IMAGE_SIZE")
    fall_class_names: str = Field(
        default="fall,fallen,lying,fall_detected,person_fall",
        validation_alias="FALL_CLASS_NAMES",
    )
    # Default OFF: a sustained 'fallen' posture (confirmed by the state machine's
    # FALL_CONFIRM_SECONDS window) is the fall signal, so no real fall is missed.
    # The posture-transition gate is an opt-in false-positive filter (it rejects
    # "already lying down" only when no prior upright was seen); robustly telling a
    # fall from a slow lie-down needs the legacy velocity-based DTS (future work).
    require_fall_transition: bool = Field(
        default=False, validation_alias="DETECTOR_REQUIRE_TRANSITION"
    )

    # ---- Fall event state machine ----------------------------------------
    fall_confirm_seconds: float = Field(default=2.0, validation_alias="FALL_CONFIRM_SECONDS")
    fall_clear_seconds: float = Field(default=3.0, validation_alias="FALL_CLEAR_SECONDS")
    alert_cooldown_seconds: float = Field(default=30.0, validation_alias="ALERT_COOLDOWN_SECONDS")
    # Live only: bridge brief real-world detection gaps so a sustained fall reads
    # as continuous evidence. Kept below FALL_CLEAR_SECONDS so recovery still works.
    evidence_hold_seconds: float = Field(default=1.0, validation_alias="EVIDENCE_HOLD_SECONDS")

    # ---- Event media ------------------------------------------------------
    save_event_snapshots: bool = Field(default=False, validation_alias="SAVE_EVENT_SNAPSHOTS")
    save_event_clips: bool = Field(default=False, validation_alias="SAVE_EVENT_CLIPS")
    events_dir: Path = Field(default=PROJECT_ROOT / "data" / "events", validation_alias="EVENTS_DIR")
    clips_dir: Path = Field(default=PROJECT_ROOT / "data" / "clips", validation_alias="CLIPS_DIR")

    # ---- Wearable ---------------------------------------------------------
    wearable_mode: WearableMode = Field(
        default=WearableMode.SIMULATION, validation_alias="WEARABLE_MODE"
    )
    wearable_device_id: str = Field(default="wearable-1", validation_alias="WEARABLE_DEVICE_ID")
    wearable_sample_seconds: float = Field(default=5.0, validation_alias="WEARABLE_SAMPLE_SECONDS")

    # ---- Alerts -----------------------------------------------------------
    webhook_url: str = Field(default="", validation_alias="WEBHOOK_URL")
    webhook_secret: str = Field(default="", validation_alias="WEBHOOK_SECRET")
    webhook_timeout_seconds: float = Field(
        default=5.0, validation_alias="WEBHOOK_TIMEOUT_SECONDS"
    )
    console_alerts_enabled: bool = Field(
        default=True, validation_alias="CONSOLE_ALERTS_ENABLED"
    )

    # ---- Monitoring loop --------------------------------------------------
    monitor_loop_interval: float = Field(
        default=0.5, validation_alias="MONITOR_LOOP_INTERVAL"
    )
    disk_warning_percent: float = Field(default=90.0, validation_alias="DISK_WARNING_PERCENT")

    # ---- Dashboard live video (privacy-sensitive; OFF by default) ----------
    # When true (development only), the dashboard exposes the live camera feed.
    # This intentionally overrides the default "no live feed" privacy posture —
    # enable only knowingly. No footage is ever written to disk regardless.
    dashboard_live_video: bool = Field(
        default=False, validation_alias="DASHBOARD_LIVE_VIDEO"
    )

    # ----------------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------------
    @field_validator("database_path", "log_dir", "events_dir", "clips_dir", mode="before")
    @classmethod
    def _blank_path_default(cls, v: Any, info: Any) -> Any:
        """A blank value in .env (e.g. ``VYTALLINK_DATABASE_PATH=``) means
        'use the default', not 'use the empty path'."""
        if v is None or str(v).strip() in ("", "."):
            defaults = {
                "database_path": PROJECT_ROOT / "data" / "database" / "vytallink.db",
                "log_dir": PROJECT_ROOT / "logs",
                "events_dir": PROJECT_ROOT / "data" / "events",
                "clips_dir": PROJECT_ROOT / "data" / "clips",
            }
            return defaults[info.field_name]
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        up = str(v).upper()
        if up not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return up

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be in 1..65535, got {v}")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {v}")
        return v

    @field_validator(
        "fall_confirm_seconds",
        "fall_clear_seconds",
        "alert_cooldown_seconds",
        "evidence_hold_seconds",
        "wearable_sample_seconds",
        "monitor_loop_interval",
        "webhook_timeout_seconds",
    )
    @classmethod
    def _validate_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"duration must be >= 0, got {v}")
        return v

    @field_validator("process_every_n_frames", "image_size")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "Settings":
        """Cross-field, production-aware validation with actionable messages."""
        problems: list[str] = []

        if self.is_production:
            if self.vision_mode == VisionMode.RTSP and not self.has_camera_target:
                problems.append(
                    "VISION_MODE=rtsp in production requires CAMERA_SOURCE (a full "
                    "RTSP URL) or CAMERA_HOST (+ optional CAMERA_PORT / "
                    "CAMERA_STREAM_PATH). See docs/hardware_needed.md."
                )
            if self.vision_mode == VisionMode.FILE and not self.camera_source:
                problems.append(
                    "VISION_MODE=file requires CAMERA_SOURCE to point at a video file."
                )
            if self.detector_mode in (DetectorMode.YOLO, DetectorMode.TENSORRT):
                if not self.model_path:
                    problems.append(
                        f"DETECTOR_MODE={self.detector_mode.value} requires MODEL_PATH "
                        "to the fall model weights. See docs/hardware_needed.md."
                    )
                elif not Path(self.model_path).expanduser().exists():
                    problems.append(
                        f"MODEL_PATH does not exist: {self.model_path!r}"
                    )

        if self.fall_clear_seconds <= 0 and self.fall_confirm_seconds <= 0:
            # Allowed (instant confirm/clear in tests) but warn-worthy; not fatal.
            pass

        if problems:
            joined = "\n  - ".join(problems)
            raise ConfigError(
                "Configuration is not valid for the selected environment:\n  - " + joined
            )
        return self

    # ----------------------------------------------------------------------
    # Convenience / derived
    # ----------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.env == Environment.DEVELOPMENT

    @property
    def simulation_active(self) -> bool:
        """True when *any* major subsystem is simulated (banner the dashboard)."""
        return (
            self.vision_mode == VisionMode.SIMULATION
            or self.detector_mode == DetectorMode.SIMULATION
            or self.wearable_mode == WearableMode.SIMULATION
        )

    @property
    def fall_class_set(self) -> set[str]:
        return {c.strip().lower() for c in self.fall_class_names.split(",") if c.strip()}

    @property
    def webhook_enabled(self) -> bool:
        return bool(self.webhook_url)

    def _rtsp_credentials(self) -> str:
        """URL-encoded ``user[:pass]@`` prefix, or "" when no username set.

        Encoding protects passwords containing ``@ : / ?`` etc. so the assembled
        RTSP URL is well-formed and the sanitizer can always strip it.
        """
        if not self.camera_username:
            return ""
        user = quote(self.camera_username, safe="")
        if self.camera_password:
            return f"{user}:{quote(self.camera_password, safe='')}@"
        return f"{user}@"

    def rtsp_url(self) -> str:
        """Assemble the credential-bearing RTSP URL, independent of VISION_MODE.

        A full URL in ``CAMERA_SOURCE`` takes precedence; otherwise the URL is
        assembled from ``CAMERA_HOST`` / ``CAMERA_PORT`` / ``CAMERA_STREAM_PATH``
        plus the separate credential fields. Used by the live pipeline and the
        camera diagnostics. **Never log this** — use :meth:`sanitized_camera_source`.
        """
        base = self.camera_source.strip()
        if base and "://" in base:
            # A full URL was supplied directly; inject creds only if absent.
            if self.camera_username and "@" not in base:
                scheme, rest = base.split("://", 1)
                return f"{scheme}://{self._rtsp_credentials()}{rest}"
            return base
        if not self.camera_host:
            return ""
        port = f":{self.camera_port}" if self.camera_port else ""
        path = self.camera_stream_path.strip()
        if path and not path.startswith("/"):
            path = "/" + path
        return f"rtsp://{self._rtsp_credentials()}{self.camera_host}{port}{path}"

    def camera_connection_string(self) -> str:
        """The effective camera connection string for the configured VISION_MODE.

        FILE mode -> the file path (``CAMERA_SOURCE``); RTSP mode -> the assembled
        :meth:`rtsp_url`. **Never log this** — use :meth:`sanitized_camera_source`.
        """
        if self.vision_mode != VisionMode.RTSP:
            return self.camera_source
        return self.rtsp_url()

    @property
    def has_camera_target(self) -> bool:
        """True when an RTSP target is configured (URL or host) / file path set."""
        if self.vision_mode == VisionMode.RTSP:
            return bool(self.camera_source.strip() or self.camera_host.strip())
        return bool(self.camera_source.strip())

    def sanitized_camera_source(self) -> str:
        return sanitize_url(self.camera_connection_string())

    def safe_summary(self) -> dict[str, Any]:
        """A log/health-safe view of configuration with all secrets redacted."""
        return {
            "env": self.env.value,
            "host": self.host,
            "port": self.port,
            "database_path": str(self.database_path),
            "log_level": self.log_level,
            "vision_mode": self.vision_mode.value,
            "camera_source": self.sanitized_camera_source(),
            "camera_username": sanitize_secret(self.camera_username),
            "camera_password": sanitize_secret(self.camera_password),
            "detector_mode": self.detector_mode.value,
            # Basename only — the absolute model path is never logged or surfaced.
            "model_file": Path(self.model_path).name if self.model_path else "(unset)",
            "confidence_threshold": self.confidence_threshold,
            "process_every_n_frames": self.process_every_n_frames,
            "image_size": self.image_size,
            "fall_confirm_seconds": self.fall_confirm_seconds,
            "fall_clear_seconds": self.fall_clear_seconds,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "save_event_snapshots": self.save_event_snapshots,
            "save_event_clips": self.save_event_clips,
            "wearable_mode": self.wearable_mode.value,
            "wearable_sample_seconds": self.wearable_sample_seconds,
            "webhook_url": sanitize_url(self.webhook_url),
            "webhook_secret": sanitize_secret(self.webhook_secret),
            "console_alerts_enabled": self.console_alerts_enabled,
        }

    def ensure_runtime_dirs(self) -> None:
        """Create the local runtime directories this configuration needs."""
        for d in (
            self.database_path.parent,
            self.log_dir,
            self.events_dir,
            self.clips_dir,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()


def load_settings(**overrides: Any) -> Settings:
    """Build a fresh Settings instance, applying keyword overrides.

    Used by tests and the simulation harness to construct deterministic
    configurations. It deliberately does NOT read the on-disk ``.env`` file
    (``_env_file=None``), so tests are isolated from a developer's local config.
    The application singleton :func:`get_settings` DOES read ``.env``.
    """
    return Settings(_env_file=None, **overrides)


def reset_settings_cache() -> None:
    """Clear the cached singleton (used in tests)."""
    get_settings.cache_clear()
