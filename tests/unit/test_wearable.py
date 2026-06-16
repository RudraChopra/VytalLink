"""Tests for the simulated wearable provider."""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.common.types import HealthStatus
from vytallink.wearable.simulated import SimulatedWearable


def test_connect_and_read(manual_clock: ManualClock):
    w = SimulatedWearable(clock=manual_clock)
    assert w.connected is False
    assert w.read() is None  # not connected yet
    w.connect()
    assert w.connected is True
    v = w.read()
    assert v is not None
    assert v.simulated is True
    assert 40 <= v.heart_rate <= 120  # plausible range
    assert 0.0 <= v.connection_quality <= 1.0
    assert v.motion >= 0.0
    assert v.device_id == "wearable-1"


def test_battery_drains_monotonically(manual_clock: ManualClock):
    w = SimulatedWearable(clock=manual_clock, battery_start=100.0, battery_drain_per_read=1.0)
    w.connect()
    first = w.read().battery
    for _ in range(5):
        last = w.read().battery
    assert last < first


def test_status_transitions(manual_clock: ManualClock):
    w = SimulatedWearable(clock=manual_clock)
    assert w.status() is HealthStatus.DISABLED
    w.connect()
    assert w.status() is HealthStatus.OK
    w.disconnect()
    assert w.status() is HealthStatus.DISABLED


def test_injected_failure_raises_but_is_isolatable(manual_clock: ManualClock):
    w = SimulatedWearable(clock=manual_clock)
    w.connect()
    w.inject_failure(1)
    with pytest.raises(ConnectionError):
        w.read()
    # Recovers on the next read.
    v = w.read()
    assert v is not None


def test_deterministic_output():
    a = SimulatedWearable(clock=ManualClock())
    b = SimulatedWearable(clock=ManualClock())
    a.connect()
    b.connect()
    seq_a = [a.read().heart_rate for _ in range(5)]
    seq_b = [b.read().heart_rate for _ in range(5)]
    assert seq_a == seq_b


def test_health_payload(manual_clock: ManualClock):
    w = SimulatedWearable(clock=manual_clock)
    w.connect()
    h = w.health()
    assert h["connected"] is True
    assert h["simulated"] is True
    assert h["name"] == "simulated"
