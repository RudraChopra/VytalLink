"""Exhaustive, deterministic tests for the fall event state machine.

All timing is driven by a ManualClock — no real sleeps.
"""

from __future__ import annotations

import itertools

import pytest

from vytallink.common.clock import ManualClock
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState, TransitionReason


def make_sm(clock: ManualClock, **over) -> FallEventStateMachine:
    counter = itertools.count(1)
    params = dict(confirm_seconds=2.0, clear_seconds=3.0, cooldown_seconds=30.0)
    params.update(over)
    return FallEventStateMachine(
        clock=clock,
        uid_factory=lambda: f"evt-{next(counter)}",
        **params,
    )


def confirm(sm: FallEventStateMachine, clock: ManualClock, conf: float = 0.9):
    """Drive the machine from NORMAL to CONFIRMED_FALL. Returns transitions."""
    out = sm.observe(True, conf)  # -> POSSIBLE
    clock.advance(sm.confirm_seconds + 0.05)
    out += sm.observe(True, conf)  # -> CONFIRMED
    return out


# --- normal / possible ----------------------------------------------------
def test_normal_activity_stays_normal(manual_clock):
    sm = make_sm(manual_clock)
    for _ in range(5):
        assert sm.observe(False, 0.0) == []
        manual_clock.advance(1.0)
    assert sm.state is FallState.NORMAL
    assert sm.current_event is None


def test_single_evidence_enters_possible(manual_clock):
    sm = make_sm(manual_clock)
    out = sm.observe(True, 0.7)
    assert len(out) == 1
    assert out[0].to_state is FallState.POSSIBLE_FALL
    assert sm.state is FallState.POSSIBLE_FALL
    assert sm.current_event.detection_count == 1


def test_brief_false_fall_is_dismissed(manual_clock):
    sm = make_sm(manual_clock)
    sm.observe(True, 0.8)  # POSSIBLE
    manual_clock.advance(0.5)  # less than confirm window
    out = sm.observe(False, 0.0)  # evidence gone
    assert out[0].to_state is FallState.NORMAL
    assert out[0].reason is TransitionReason.BRIEF_EVIDENCE_DISMISSED
    assert sm.current_event is None
    # No alert ever fired.
    assert all(not t.alert for t in out)


# --- confirmation timing --------------------------------------------------
def test_not_confirmed_before_confirm_window(manual_clock):
    sm = make_sm(manual_clock, confirm_seconds=2.0)
    sm.observe(True, 0.8)
    manual_clock.advance(1.99)
    out = sm.observe(True, 0.85)
    assert out == []  # still accumulating
    assert sm.state is FallState.POSSIBLE_FALL


def test_confirmed_at_confirm_window(manual_clock):
    sm = make_sm(manual_clock, confirm_seconds=2.0)
    sm.observe(True, 0.8)
    manual_clock.advance(2.0)
    out = sm.observe(True, 0.95)
    assert len(out) == 1
    assert out[0].to_state is FallState.CONFIRMED_FALL
    assert out[0].reason is TransitionReason.SUSTAINED_EVIDENCE
    assert out[0].alert is True  # first event always alerts
    assert sm.current_event.confirmed_time is not None


def test_highest_confidence_tracked(manual_clock):
    sm = make_sm(manual_clock)
    sm.observe(True, 0.6)
    manual_clock.advance(2.1)
    sm.observe(True, 0.91)
    manual_clock.advance(0.1)
    sm.observe(True, 0.72)  # lower; should not reduce the max
    assert sm.current_event.highest_confidence == pytest.approx(0.91)


# --- exactly one alert + duplicate suppression ----------------------------
def test_exactly_one_alert_per_event(manual_clock):
    sm = make_sm(manual_clock)
    out = confirm(sm, manual_clock)
    alerts = [t for t in out if t.alert]
    assert len(alerts) == 1
    # Repeated sustained evidence must not produce more alert transitions.
    extra_alerts = 0
    for _ in range(10):
        manual_clock.advance(0.1)
        for t in sm.observe(True, 0.95):
            extra_alerts += int(t.alert)
    assert extra_alerts == 0
    assert sm.state is FallState.CONFIRMED_FALL
    assert sm.current_event.detection_count >= 11


# --- recovery timing ------------------------------------------------------
def test_recovery_then_resolution(manual_clock):
    sm = make_sm(manual_clock, clear_seconds=3.0)
    confirm(sm, manual_clock)
    out = sm.observe(False, 0.0)  # evidence cleared
    assert out[0].to_state is FallState.RECOVERING
    manual_clock.advance(2.99)
    assert sm.observe(False, 0.0) == []  # still within clear window
    manual_clock.advance(0.02)
    out = sm.observe(False, 0.0)
    assert out[0].to_state is FallState.RESOLVED
    assert out[0].reason is TransitionReason.RECOVERY_TIMEOUT
    assert sm.current_event.resolved_time is not None


def test_recovery_cancelled_by_returning_evidence(manual_clock):
    sm = make_sm(manual_clock)
    confirm(sm, manual_clock)
    sm.observe(False, 0.0)  # RECOVERING
    assert sm.state is FallState.RECOVERING
    out = sm.observe(True, 0.8)  # person still down
    assert out[0].to_state is FallState.CONFIRMED_FALL
    assert out[0].reason is TransitionReason.EVIDENCE_RETURNED
    assert out[0].alert is False  # same event, no new alert
    assert sm.current_event.recovering_since is None


# --- cooldown / second event ----------------------------------------------
def _resolve_to_normal(sm, clock):
    sm.observe(False, 0.0)  # CONFIRMED -> RECOVERING
    clock.advance(sm.clear_seconds + 0.05)
    sm.observe(False, 0.0)  # -> RESOLVED
    sm.observe(False, 0.0)  # RESOLVED -> NORMAL


def test_second_event_within_cooldown_suppresses_alert(manual_clock):
    sm = make_sm(manual_clock, cooldown_seconds=30.0)
    out1 = confirm(sm, manual_clock)
    assert any(t.alert for t in out1)
    _resolve_to_normal(sm, manual_clock)
    # Only a few seconds have passed (< 30s cooldown).
    out2 = confirm(sm, manual_clock)
    assert sm.state is FallState.CONFIRMED_FALL
    assert all(not t.alert for t in out2)  # alert suppressed by cooldown
    # A new, distinct event was created.
    assert sm.current_event.event_uid != out1[0].event_uid


def test_second_event_after_cooldown_alerts(manual_clock):
    sm = make_sm(manual_clock, cooldown_seconds=30.0)
    confirm(sm, manual_clock)
    _resolve_to_normal(sm, manual_clock)
    manual_clock.advance(31.0)  # cooldown elapsed
    out2 = confirm(sm, manual_clock)
    assert sum(int(t.alert) for t in out2) == 1


def test_new_event_directly_from_resolved_on_evidence(manual_clock):
    sm = make_sm(manual_clock)
    confirm(sm, manual_clock)
    sm.observe(False, 0.0)  # RECOVERING
    manual_clock.advance(3.1)
    sm.observe(False, 0.0)  # RESOLVED
    out = sm.observe(True, 0.8)  # evidence again -> new event, POSSIBLE
    assert out[0].reason is TransitionReason.NEW_EVENT_AFTER_RESOLVED
    assert out[0].to_state is FallState.POSSIBLE_FALL


# --- manual operations ----------------------------------------------------
def test_manual_resolve(manual_clock):
    sm = make_sm(manual_clock)
    confirm(sm, manual_clock)
    out = sm.manual_resolve()
    assert out[0].to_state is FallState.RESOLVED
    assert out[0].reason is TransitionReason.MANUAL_RESOLVE
    # Resolving when already normal is a no-op.
    sm.observe(False, 0.0)  # -> NORMAL
    assert sm.manual_resolve() == []


def test_reset_returns_to_normal(manual_clock):
    sm = make_sm(manual_clock)
    confirm(sm, manual_clock)
    out = sm.reset()
    assert out[0].to_state is FallState.NORMAL
    assert sm.state is FallState.NORMAL
    assert sm.current_event is None
    # After reset, cooldown is cleared: next confirm alerts immediately.
    out2 = confirm(sm, manual_clock)
    assert sum(int(t.alert) for t in out2) == 1


def test_zero_confirm_window_confirms_on_second_sample(manual_clock):
    # Edge: instant confirmation when confirm_seconds == 0.
    sm = make_sm(manual_clock, confirm_seconds=0.0)
    out = sm.observe(True, 0.9)  # POSSIBLE
    assert out[0].to_state is FallState.POSSIBLE_FALL
    out = sm.observe(True, 0.9)  # elapsed 0 >= 0 -> CONFIRMED
    assert out[0].to_state is FallState.CONFIRMED_FALL
