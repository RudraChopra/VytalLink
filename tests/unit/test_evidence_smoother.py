"""Tests for FallEvidenceSmoother (gap-bridging, ManualClock)."""

from __future__ import annotations

import pytest

from vytallink.common.clock import ManualClock
from vytallink.vision.evidence import FallEvidenceSmoother


def test_fallen_is_evidence_and_arms_hold():
    clock = ManualClock()
    s = FallEvidenceSmoother(1.0, clock=clock)
    ev, conf = s.update(True, 0.9, had_detection=True, had_upright=False)
    assert ev is True and conf == 0.9 and s.holding


def test_bridges_brief_detection_gap():
    clock = ManualClock()
    s = FallEvidenceSmoother(1.0, clock=clock)
    s.update(True, 0.9, had_detection=True, had_upright=False)   # fallen
    clock.advance(0.4)
    ev, conf = s.update(False, 0.0, had_detection=False, had_upright=False)  # gap
    assert ev is True and conf == 0.9   # bridged with last confidence


def test_hold_expires_after_window():
    clock = ManualClock()
    s = FallEvidenceSmoother(1.0, clock=clock)
    s.update(True, 0.9, had_detection=True, had_upright=False)
    clock.advance(1.2)  # past the 1.0s hold
    ev, _ = s.update(False, 0.0, had_detection=False, had_upright=False)
    assert ev is False and not s.holding


def test_upright_cancels_hold_immediately():
    clock = ManualClock()
    s = FallEvidenceSmoother(1.0, clock=clock)
    s.update(True, 0.9, had_detection=True, had_upright=False)
    clock.advance(0.2)
    # Person clearly stood up within the hold window -> evidence ends now.
    ev, _ = s.update(False, 0.0, had_detection=True, had_upright=True)
    assert ev is False and not s.holding


def test_sparse_fallen_sequence_stays_continuous():
    """F · F · F (sparse) should read as continuous evidence."""
    clock = ManualClock()
    s = FallEvidenceSmoother(1.0, clock=clock)
    seq = [(True, True), (False, False), (True, True), (False, False), (True, True)]
    results = []
    for raw, _ in seq:
        ev, _ = s.update(raw, 0.8 if raw else 0.0, had_detection=raw, had_upright=False)
        results.append(ev)
        clock.advance(0.3)
    assert all(results)  # never dropped to False across the gaps
