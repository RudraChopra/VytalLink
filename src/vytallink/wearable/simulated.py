"""Deterministic simulated wearable.

A *real* working provider labeled "simulation". It produces realistic but
clearly-simulated vitals:

* heart rate — a smooth baseline with deterministic sinusoidal variation,
* motion     — a small activity value with periodic peaks,
* battery    — slowly draining from a configurable start level,
* connection quality — high with mild deterministic variation.

Output is deterministic (driven by an internal step counter + ``math.sin``,
no RNG) so tests are reproducible. Vital timestamps use a real UTC clock so the
dashboard shows live-updating freshness independent of the fall-simulation
clock. A failure can be injected to exercise graceful degradation.
"""

from __future__ import annotations

import math

from vytallink.common.clock import Clock, SystemClock
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import VitalReading
from vytallink.wearable.base import WearableProvider

log = get_logger("wearable.simulated")


class SimulatedWearable(WearableProvider):
    name = "simulated"

    def __init__(
        self,
        device_id: str = "wearable-1",
        display_name: str = "Simulated wearable",
        *,
        clock: Clock | None = None,
        hr_baseline: float = 72.0,
        battery_start: float = 100.0,
        battery_drain_per_read: float = 0.05,
    ) -> None:
        super().__init__(device_id, display_name)
        self.clock: Clock = clock or SystemClock()
        self.hr_baseline = hr_baseline
        self.battery = battery_start
        self.battery_drain_per_read = battery_drain_per_read
        self._step = 0
        self._fail_remaining = 0

    def inject_failure(self, count: int) -> None:
        """Make the next ``count`` reads fail (simulate a device dropout)."""
        self._fail_remaining = max(0, int(count))

    def connect(self) -> None:
        if self._fail_remaining > 0:
            self._connected = False
            self._last_error = "simulated connect failure"
            raise ConnectionError(self._last_error)
        self._connected = True
        self._last_error = None
        log.info("Simulated wearable %s connected", self.device_id)

    def read(self) -> VitalReading | None:
        if not self._connected:
            return None
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            self._last_error = "simulated read failure"
            raise ConnectionError(self._last_error)

        self._step += 1
        s = self._step
        # Heart rate: baseline + slow sinusoid + faster small ripple.
        hr = self.hr_baseline + 6.0 * math.sin(s / 12.0) + 2.0 * math.sin(s / 3.0)
        # Motion: mostly low, with a periodic activity bump.
        motion = round(0.15 + 0.10 * abs(math.sin(s / 7.0)) + (0.4 if s % 17 == 0 else 0.0), 3)
        # Connection quality: high, mild variation.
        cq = round(0.92 + 0.05 * (0.5 + 0.5 * math.sin(s / 5.0)), 3)
        # Battery: monotonically draining, clamped at 5% floor for simulation.
        self.battery = max(5.0, self.battery - self.battery_drain_per_read)

        self._last_error = None
        return VitalReading(
            timestamp=self.clock.now(),
            device_id=self.device_id,
            heart_rate=round(hr, 1),
            motion=motion,
            connection_quality=cq,
            battery=round(self.battery, 1),
            simulated=True,
            metadata={"step": s, "source": "simulation"},
        )
