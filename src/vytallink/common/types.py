"""Shared value types passed between layers (camera → detector → events)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    """Coarse health classification used across subsystems and the API."""

    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class ProviderMode(str, Enum):
    """How a provider is sourcing its data."""

    SIMULATION = "simulation"
    FILE = "file"
    RTSP = "rtsp"
    HARDWARE = "hardware"
    DISABLED = "disabled"


@dataclass(slots=True)
class Frame:
    """A single captured frame (or a simulated stand-in for one).

    In simulation mode ``image`` is ``None`` — no pixel data is produced, which
    keeps the simulation lightweight and avoids any privacy surface. Real camera
    providers populate ``image`` with a numpy array.
    """

    frame_id: int
    timestamp: datetime
    source_id: str
    width: int = 0
    height: int = 0
    image: Any | None = None  # numpy.ndarray when from a real camera

    @property
    def has_image(self) -> bool:
        return self.image is not None


@dataclass(slots=True)
class RawDetection:
    """A single raw detection emitted by a detector for one frame.

    A raw detection is *not* a confirmed fall — it is one observation that the
    state machine aggregates over time.
    """

    timestamp: datetime
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    source_id: str
    frame_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VitalReading:
    """A single wearable vital reading."""

    timestamp: datetime
    device_id: str
    heart_rate: float | None
    motion: float | None
    connection_quality: float | None = None
    battery: float | None = None
    simulated: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
