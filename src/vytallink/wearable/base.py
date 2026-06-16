"""Wearable provider interface and connection state.

The real wearable has NOT been selected, so Phase 1 ships only a deterministic
simulated provider. This interface is what a future BLE / vendor-API wearable
must implement. See docs/hardware_needed.md for what is required to connect a
real device.
"""

from __future__ import annotations

import abc
from typing import Any

from vytallink.common.logging_setup import get_logger
from vytallink.common.types import HealthStatus, VitalReading

log = get_logger("wearable")


class WearableProvider(abc.ABC):
    name: str = "base"
    device_type: str = "wearable"

    def __init__(self, device_id: str = "wearable-1", display_name: str = "Wearable") -> None:
        self.device_id = device_id
        self.display_name = display_name
        self._connected = False
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def simulated(self) -> bool:
        return True

    @abc.abstractmethod
    def connect(self) -> None:  # pragma: no cover - interface
        ...

    @abc.abstractmethod
    def read(self) -> VitalReading | None:  # pragma: no cover - interface
        """Return the latest vital reading, or None if unavailable."""
        ...

    def disconnect(self) -> None:
        self._connected = False

    def status(self) -> HealthStatus:
        if not self._connected:
            return HealthStatus.DOWN if self._last_error else HealthStatus.DISABLED
        return HealthStatus.OK

    def health(self) -> dict[str, Any]:
        return {
            "status": self.status().value,
            "device_id": self.device_id,
            "name": self.name,
            "connected": self._connected,
            "simulated": self.simulated,
            "last_error": self._last_error,
        }
