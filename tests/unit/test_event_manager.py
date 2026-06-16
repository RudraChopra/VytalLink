"""Tests for EventManager: persistence, exactly-one-alert, labeling, resolution.

Uses a deterministic fake dispatcher (records to the alerts table like the real
one) so we can assert DB state without console/network providers.
"""

from __future__ import annotations

import itertools

import pytest

from vytallink.alerts.base import AlertEvent, AlertResult
from vytallink.common.clock import ManualClock, isoformat
from vytallink.database.models import AlertRow
from vytallink.events.manager import EventManager
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState, HumanLabel


class FakeDispatcher:
    """Records each dispatch to the alerts table and counts calls."""

    def __init__(self, repos, clock):
        self.repos = repos
        self.clock = clock
        self.calls: list[AlertEvent] = []

    async def dispatch(self, alert: AlertEvent) -> list[AlertResult]:
        self.calls.append(alert)
        self.repos.alerts.record(
            AlertRow(
                event_uid=alert.event_uid,
                provider="fake",
                attempt_time=isoformat(alert.timestamp),
                success=True,
                response_metadata={"simulated": alert.simulated},
            )
        )
        return [
            AlertResult(
                provider="fake", success=True, attempt_time=self.clock.now()
            )
        ]


def build_manager(repos, clock: ManualClock, **over) -> tuple[EventManager, FakeDispatcher]:
    counter = itertools.count(1)
    sm = FallEventStateMachine(
        confirm_seconds=over.get("confirm_seconds", 2.0),
        clear_seconds=over.get("clear_seconds", 3.0),
        cooldown_seconds=over.get("cooldown_seconds", 30.0),
        clock=clock,
        uid_factory=lambda: f"evt-{next(counter)}",
    )
    dispatcher = FakeDispatcher(repos, clock)
    mgr = EventManager(repos, sm, dispatcher, clock=clock, simulated=True)
    return mgr, dispatcher


async def drive_confirm(mgr, clock, conf=0.92):
    await mgr.observe(True, conf)
    clock.advance(mgr.sm.confirm_seconds + 0.05)
    return await mgr.observe(True, conf)


@pytest.mark.asyncio
async def test_confirmed_fall_creates_one_event_and_one_alert(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    await drive_confirm(mgr, manual_clock)
    events = repos.events.list()
    assert len(events) == 1
    assert events[0].state == FallState.CONFIRMED_FALL.value
    assert events[0].confirmed_time is not None
    assert len(disp.calls) == 1
    assert repos.alerts.count() == 1


@pytest.mark.asyncio
async def test_possible_blip_not_persisted(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    await mgr.observe(True, 0.8)  # POSSIBLE
    manual_clock.advance(0.5)
    await mgr.observe(False, 0.0)  # dismissed
    assert repos.events.count() == 0
    assert len(disp.calls) == 0


@pytest.mark.asyncio
async def test_duplicate_evidence_no_duplicate_alert(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock)
    await drive_confirm(mgr, manual_clock)
    for _ in range(8):
        manual_clock.advance(0.2)
        await mgr.observe(True, 0.95)
    assert repos.events.count() == 1
    assert len(disp.calls) == 1
    assert repos.alerts.count() == 1


@pytest.mark.asyncio
async def test_recovery_resolution_updates_same_row(repos, manual_clock):
    mgr, _ = build_manager(repos, manual_clock, clear_seconds=3.0)
    await drive_confirm(mgr, manual_clock)
    uid = repos.events.list()[0].event_uid
    await mgr.observe(False, 0.0)  # RECOVERING
    assert repos.events.get_by_uid(uid).state == FallState.RECOVERING.value
    manual_clock.advance(3.1)
    await mgr.observe(False, 0.0)  # RESOLVED
    row = repos.events.get_by_uid(uid)
    assert row.state == FallState.RESOLVED.value
    assert row.resolved_time is not None
    assert repos.events.count() == 1  # still one row


@pytest.mark.asyncio
async def test_label_event(repos, manual_clock):
    mgr, _ = build_manager(repos, manual_clock)
    await drive_confirm(mgr, manual_clock)
    uid = repos.events.list()[0].event_uid
    row = await mgr.label_event(uid, HumanLabel.REAL_FALL.value)
    assert row.human_label == "real_fall"
    with pytest.raises(ValueError):
        await mgr.label_event(uid, "bogus")


@pytest.mark.asyncio
async def test_manual_resolve_live_event(repos, manual_clock):
    mgr, _ = build_manager(repos, manual_clock)
    await drive_confirm(mgr, manual_clock)
    uid = repos.events.list()[0].event_uid
    row = await mgr.resolve_event(uid, note="checked, false alarm")
    assert row.state == FallState.RESOLVED.value
    assert row.resolution_note == "checked, false alarm"
    assert mgr.current_state() is FallState.RESOLVED


@pytest.mark.asyncio
async def test_second_event_after_cooldown_two_events_two_alerts(repos, manual_clock):
    mgr, disp = build_manager(repos, manual_clock, cooldown_seconds=30.0)
    await drive_confirm(mgr, manual_clock)
    # resolve first
    await mgr.observe(False, 0.0)
    manual_clock.advance(3.1)
    await mgr.observe(False, 0.0)  # RESOLVED
    await mgr.observe(False, 0.0)  # NORMAL
    manual_clock.advance(31.0)  # past cooldown
    await drive_confirm(mgr, manual_clock)
    assert repos.events.count() == 2
    assert len(disp.calls) == 2
    assert repos.alerts.count() == 2


@pytest.mark.asyncio
async def test_reset_clears_state(repos, manual_clock):
    mgr, _ = build_manager(repos, manual_clock)
    await drive_confirm(mgr, manual_clock)
    await mgr.reset()
    assert mgr.current_state() is FallState.NORMAL
