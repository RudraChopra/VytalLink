"""Factory for the configured wearable provider."""

from __future__ import annotations

from vytallink.common.clock import Clock
from vytallink.config import Settings, WearableMode
from vytallink.wearable.base import WearableProvider
from vytallink.wearable.simulated import SimulatedWearable


def build_wearable(settings: Settings, clock: Clock | None = None) -> WearableProvider:
    if settings.wearable_mode == WearableMode.SIMULATION:
        return SimulatedWearable(device_id=settings.wearable_device_id, clock=clock)
    raise ValueError(f"Unknown wearable mode: {settings.wearable_mode}")  # pragma: no cover
