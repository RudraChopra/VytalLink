"""The fall-event state machine.

This is pure, side-effect-free logic over an injected :class:`Clock`. It does
not touch the database or dispatch alerts directly — instead :meth:`observe`
returns :class:`Transition` records describing what happened, and a
``Transition`` with ``alert=True`` tells the caller (``EventManager``) to
dispatch exactly one alert. This keeps the timing/cooldown/duplicate logic
fully unit-testable without sleeps, databases, or network.

Timing uses ``clock.monotonic()`` for durations and ``clock.now()`` for stored
timestamps. Tests and the simulation driver use a ``ManualClock`` to advance
time deterministically.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from vytallink.common.clock import Clock, SystemClock
from vytallink.common.logging_setup import get_logger
from vytallink.events.states import FallState, Transition, TransitionReason

log = get_logger("events.state_machine")


def _default_uid_factory() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class FallEvent:
    """In-memory representation of the current event being tracked."""

    event_uid: str
    state: FallState
    start_time: datetime
    source_device: str
    event_type: str = "fall"
    confirmed_time: datetime | None = None
    end_time: datetime | None = None
    resolved_time: datetime | None = None
    highest_confidence: float = 0.0
    detection_count: int = 0
    alerted: bool = False
    # Internal monotonic timers (not persisted).
    possible_since: float = 0.0
    recovering_since: float | None = None


class FallEventStateMachine:
    """Aggregates per-frame evidence into confirmed fall events."""

    def __init__(
        self,
        *,
        confirm_seconds: float,
        clear_seconds: float,
        cooldown_seconds: float,
        source_device: str = "camera-1",
        clock: Clock | None = None,
        uid_factory: Callable[[], str] | None = None,
    ) -> None:
        self.confirm_seconds = float(confirm_seconds)
        self.clear_seconds = float(clear_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.source_device = source_device
        self.clock: Clock = clock or SystemClock()
        self._uid_factory = uid_factory or _default_uid_factory

        self._state: FallState = FallState.NORMAL
        self._event: FallEvent | None = None
        self._last_alert_mono: float | None = None

    # -- introspection -----------------------------------------------------
    @property
    def state(self) -> FallState:
        return self._state

    @property
    def current_event(self) -> FallEvent | None:
        return self._event

    def snapshot(self) -> dict:
        """A status-friendly view of the current state/event."""
        ev = self._event
        return {
            "state": self._state.value,
            "event_uid": ev.event_uid if ev else None,
            "highest_confidence": round(ev.highest_confidence, 4) if ev else 0.0,
            "detection_count": ev.detection_count if ev else 0,
            "start_time": ev.start_time if ev else None,
            "confirmed_time": ev.confirmed_time if ev else None,
            "end_time": ev.end_time if ev else None,
            "resolved_time": ev.resolved_time if ev else None,
        }

    # -- core --------------------------------------------------------------
    def observe(
        self,
        evidence: bool,
        confidence: float = 0.0,
        *,
        now: datetime | None = None,
        mono: float | None = None,
    ) -> list[Transition]:
        """Feed one observation. Returns any transitions that occurred.

        Args:
            evidence: Whether this observation constitutes fall evidence
                (i.e. a fall-class detection at/above the confidence threshold).
            confidence: The confidence of the strongest fall detection (0..1).
            now: Optional explicit wall-clock timestamp (defaults to clock).
            mono: Optional explicit monotonic value (defaults to clock).
        """
        now = now or self.clock.now()
        mono = self.clock.monotonic() if mono is None else mono
        confidence = max(0.0, min(1.0, float(confidence)))

        handler = {
            FallState.NORMAL: self._on_normal,
            FallState.POSSIBLE_FALL: self._on_possible,
            FallState.CONFIRMED_FALL: self._on_confirmed,
            FallState.RECOVERING: self._on_recovering,
            FallState.RESOLVED: self._on_resolved,
        }[self._state]

        transitions = handler(evidence, confidence, now, mono)
        for t in transitions:
            log.info("FALL STATE %s", t.describe())
        return transitions

    # -- per-state handlers ------------------------------------------------
    def _start_event(self, confidence: float, now: datetime, mono: float) -> FallEvent:
        return FallEvent(
            event_uid=self._uid_factory(),
            state=FallState.POSSIBLE_FALL,
            start_time=now,
            source_device=self.source_device,
            highest_confidence=confidence,
            detection_count=1,
            possible_since=mono,
        )

    def _transition(
        self,
        to_state: FallState,
        reason: TransitionReason,
        now: datetime,
        *,
        alert: bool = False,
    ) -> Transition:
        ev = self._event
        from_state = self._state
        self._state = to_state
        if ev is not None:
            ev.state = to_state
        return Transition(
            timestamp=now,
            from_state=from_state,
            to_state=to_state,
            event_uid=ev.event_uid if ev else "",
            reason=reason,
            alert=alert,
            highest_confidence=ev.highest_confidence if ev else 0.0,
            detection_count=ev.detection_count if ev else 0,
        )

    def _on_normal(self, evidence, confidence, now, mono) -> list[Transition]:
        if not evidence:
            return []
        self._event = self._start_event(confidence, now, mono)
        return [self._transition(FallState.POSSIBLE_FALL, TransitionReason.EVIDENCE_DETECTED, now)]

    def _on_possible(self, evidence, confidence, now, mono) -> list[Transition]:
        ev = self._event
        assert ev is not None
        if not evidence:
            # Brief, unsustained evidence — dismiss and return to NORMAL.
            self._event = None
            return [
                self._transition(
                    FallState.NORMAL, TransitionReason.BRIEF_EVIDENCE_DISMISSED, now
                )
            ]
        ev.detection_count += 1
        ev.highest_confidence = max(ev.highest_confidence, confidence)
        if (mono - ev.possible_since) >= self.confirm_seconds:
            ev.confirmed_time = now
            should_alert = self._cooldown_ok(mono)
            if should_alert:
                self._last_alert_mono = mono
                ev.alerted = True
            return [
                self._transition(
                    FallState.CONFIRMED_FALL,
                    TransitionReason.SUSTAINED_EVIDENCE,
                    now,
                    alert=should_alert,
                )
            ]
        return []  # still accumulating; no state change

    def _on_confirmed(self, evidence, confidence, now, mono) -> list[Transition]:
        ev = self._event
        assert ev is not None
        if evidence:
            # Sustained / repeated evidence: stay confirmed, no duplicate alert.
            ev.detection_count += 1
            ev.highest_confidence = max(ev.highest_confidence, confidence)
            return []
        # Evidence cleared: begin recovery window.
        ev.recovering_since = mono
        ev.end_time = now
        return [self._transition(FallState.RECOVERING, TransitionReason.EVIDENCE_CLEARED, now)]

    def _on_recovering(self, evidence, confidence, now, mono) -> list[Transition]:
        ev = self._event
        assert ev is not None
        if evidence:
            # Person still down — cancel recovery, back to confirmed (same event).
            ev.detection_count += 1
            ev.highest_confidence = max(ev.highest_confidence, confidence)
            ev.recovering_since = None
            ev.end_time = None
            return [
                self._transition(
                    FallState.CONFIRMED_FALL, TransitionReason.EVIDENCE_RETURNED, now
                )
            ]
        since = ev.recovering_since if ev.recovering_since is not None else mono
        if (mono - since) >= self.clear_seconds:
            ev.resolved_time = now
            if ev.end_time is None:
                ev.end_time = now
            return [
                self._transition(
                    FallState.RESOLVED, TransitionReason.RECOVERY_TIMEOUT, now
                )
            ]
        return []  # still within recovery window

    def _on_resolved(self, evidence, confidence, now, mono) -> list[Transition]:
        if evidence:
            # A brand-new, independent event begins.
            prev = self._event
            self._event = self._start_event(confidence, now, mono)
            from_state = FallState.RESOLVED
            self._state = FallState.POSSIBLE_FALL
            self._event.state = FallState.POSSIBLE_FALL
            return [
                Transition(
                    timestamp=now,
                    from_state=from_state,
                    to_state=FallState.POSSIBLE_FALL,
                    event_uid=self._event.event_uid,
                    reason=TransitionReason.NEW_EVENT_AFTER_RESOLVED,
                    alert=False,
                    highest_confidence=self._event.highest_confidence,
                    detection_count=self._event.detection_count,
                )
            ]
        # No evidence: clear the finished event and return to NORMAL.
        self._event = None
        return [self._transition(FallState.NORMAL, TransitionReason.READY_AFTER_RESOLVED, now)]

    # -- manual operations -------------------------------------------------
    def manual_resolve(self, *, now: datetime | None = None) -> list[Transition]:
        """Force the live event to RESOLVED (caregiver action)."""
        if self._state in (FallState.NORMAL, FallState.RESOLVED):
            return []
        now = now or self.clock.now()
        ev = self._event
        if ev is not None:
            ev.resolved_time = now
            if ev.end_time is None:
                ev.end_time = now
        return [self._transition(FallState.RESOLVED, TransitionReason.MANUAL_RESOLVE, now)]

    def reset(self, *, now: datetime | None = None) -> list[Transition]:
        """Reset the machine to NORMAL, discarding the in-memory event."""
        now = now or self.clock.now()
        from_state = self._state
        self._event = None
        self._state = FallState.NORMAL
        self._last_alert_mono = None
        return [
            Transition(
                timestamp=now,
                from_state=from_state,
                to_state=FallState.NORMAL,
                event_uid="",
                reason=TransitionReason.RESET,
            )
        ]

    # -- helpers -----------------------------------------------------------
    def _cooldown_ok(self, mono: float) -> bool:
        if self._last_alert_mono is None:
            return True
        return (mono - self._last_alert_mono) >= self.cooldown_seconds
