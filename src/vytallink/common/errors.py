"""Project exception hierarchy for consistent error handling."""

from __future__ import annotations


class VytalLinkError(Exception):
    """Base class for all VytalLink errors."""


class ConfigError(VytalLinkError):
    """Raised when configuration is missing or invalid.

    Carries a human-readable, actionable message safe to show at startup.
    """


class DatabaseError(VytalLinkError):
    """Raised for database initialization or query failures."""


class NotFoundError(VytalLinkError):
    """Raised when a requested resource (e.g. event) does not exist."""


class ProviderError(VytalLinkError):
    """Base class for provider (camera/detector/wearable/alert) failures."""


class CameraError(ProviderError):
    """Raised for camera open/read failures."""


class DetectorError(ProviderError):
    """Raised for detector load/inference failures."""


class AlertDeliveryError(ProviderError):
    """Raised when an alert provider fails to deliver. Never crashes the app —
    it is caught at the dispatcher boundary and recorded."""
