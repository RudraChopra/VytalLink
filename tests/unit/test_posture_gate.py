"""Tests for the DTS-lite PostureTransitionGate (deterministic, ManualClock)."""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.vision.posture_gate import GateConfig, PostureTransitionGate


def _gate(clock):
    return PostureTransitionGate(
        GateConfig(min_upright_frames=3, min_fallen_frames=3, transition_window_seconds=2.5,
                   stale_seconds=2.0),
        clock=clock,
    )


def _feed(gate, clock, *, fallen=0.0, upright=0.0, has=True, dt=0.1):
    res = gate.observe(fallen_conf=fallen, upright_conf=upright, has_detection=has)
    clock.advance(dt)
    return res


def test_upright_then_fallen_transition_confirms():
    clock = ManualClock()
    g = _gate(clock)
    # 3 sustained standing frames establish "was upright".
    for _ in range(3):
        assert _feed(g, clock, upright=0.9) is False
    # 3 sustained fallen frames within the window confirm the transition.
    assert _feed(g, clock, fallen=0.8) is False  # fallen_run=1
    assert _feed(g, clock, fallen=0.8) is False  # fallen_run=2
    assert _feed(g, clock, fallen=0.8) is True   # fallen_run=3 -> confirmed


def test_already_lying_never_triggers():
    """Someone on the floor from the start (no preceding upright) is not a fall."""
    clock = ManualClock()
    g = _gate(clock)
    for _ in range(10):
        assert _feed(g, clock, fallen=0.9) is False


def test_standing_up_clears_active_fall():
    clock = ManualClock()
    g = _gate(clock)
    for _ in range(3):
        _feed(g, clock, upright=0.9)
    for _ in range(3):
        last = _feed(g, clock, fallen=0.85)
    assert last is True
    # Now the person stands up -> fall no longer active.
    assert _feed(g, clock, upright=0.9) is False
    assert g.fall_active is False


def test_too_slow_transition_outside_window_does_not_confirm():
    clock = ManualClock()
    g = _gate(clock)
    for _ in range(3):
        _feed(g, clock, upright=0.9)
    # A long ambiguous gap (>2.5s) before fallen frames: with detections present
    # but neither clearly upright nor fallen, the upright memory ages out.
    for _ in range(4):
        _feed(g, clock, fallen=0.0, upright=0.0, has=True, dt=1.0)  # ambiguous, 4s total
    res = False
    for _ in range(3):
        res = _feed(g, clock, fallen=0.85)
    assert res is False  # transition window (2.5s) exceeded


def test_stale_gap_resets_memory():
    clock = ManualClock()
    g = _gate(clock)
    for _ in range(3):
        _feed(g, clock, upright=0.9)
    # No detection at all for > stale_seconds wipes the upright memory.
    _feed(g, clock, has=False, dt=2.5)
    res = False
    for _ in range(3):
        res = _feed(g, clock, fallen=0.9)
    assert res is False  # treated as already-lying after the stale reset


def test_low_confidence_fallen_does_not_count():
    clock = ManualClock()
    g = _gate(clock)
    for _ in range(3):
        _feed(g, clock, upright=0.9)
    # fallen confidence below the gate threshold (0.60) never builds a fallen run.
    res = False
    for _ in range(5):
        res = _feed(g, clock, fallen=0.45)
    assert res is False
