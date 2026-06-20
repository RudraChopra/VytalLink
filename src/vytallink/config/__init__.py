"""Environment-based configuration for VytalLink."""

from vytallink.config.cameras import CameraConfig, cameras_from_env
from vytallink.config.settings import (
    PROJECT_ROOT,
    DetectorMode,
    Environment,
    Settings,
    VisionMode,
    WearableMode,
    get_settings,
    load_settings,
    reset_settings_cache,
)

__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "Environment",
    "VisionMode",
    "DetectorMode",
    "WearableMode",
    "CameraConfig",
    "cameras_from_env",
    "get_settings",
    "load_settings",
    "reset_settings_cache",
]
