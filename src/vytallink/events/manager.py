"""EventManager: bridges the state machine to persistence and alerting.

Responsibilities:

* Run observations through :class:`FallEventStateMachine`.
* Persist confirmed events and their lifecycle transitions to the database
  (POSSIBLE blips that never confirm are intentionally *not* persisted).
* Dispatch exactly one alert when an event confirms (the state machine decides
  this; the manager just acts on ``transition.alert``).
* Provide caregiver operations: label and resolve.

It is async because alert dispatch (webhook) is async. The state machine itself
remains synchronous and independently testable.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from vytallink.alerts.base import AlertDispatcherProtocol, AlertEvent
from vytallink.common.clock import Clock, SystemClock, isoformat
from vytallink.common.logging_setup import get_logger
from vytallink.database.models import EventRow
from vytallink.database.repositories import Repositories
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState, HumanLabel, Transition

log = get_logger("events.manager")


class EventManager:
    #: event_type stamped on events created while synthetic fall testing is active,
    #: so they are explicitly distinguishable from real falls (not by confidence).
    SYNTHETIC_EVENT_TYPE = "fall_synthetic"

    def __init__(
        self,
        repos: Repositories,
        state_machine: FallEventStateMachine,
        dispatcher: AlertDispatcherProtocol | None = None,
        *,
        clock: Clock | None = None,
        simulated: bool = True,
        synthetic: bool = False,
        snapshot_fn: Callable[[Any], None] | None = None,
    ) -> None:
        self.repos = repos
        self.sm = state_machine
        self.dispatcher = dispatcher
        self.clock: Clock = clock or SystemClock()
        self.simulated = simulated
        # When True, persisted events are tagged event_type='fall_synthetic' so a
        # forced/validation fall is never mistaken for a real one.
        self.synthetic = synthetic
        # Called exactly once, with the new event, when an incident is first
        # CONFIRMED — to persist a vitals snapshot. Failure-isolated here AND in
        # the callback, so a snapshot error can never break observe/persistence.
        self._snapshot_fn = snapshot_fn
        self._lock = asyncio.Lock()
        self.last_alert_results: list = []

    # -- observation -------------------------------------------------------
    async def observe(
        self, evidence: bool, confidence: float = 0.0, **clock_overrides: Any
    ) -> list[Transition]:
        """Observe one evidence sample; persist + alert as needed."""
        async with self._lock:
            transitions = self.sm.observe(evidence, confidence, **clock_overrides)
            for t in transitions:
                self._persist_transition(t)
                if t.alert:
                    await self._dispatch_alert(t)
            return transitions

    def _persist_transition(self, t: Transition) -> None:
        """Create or update the event row for active-state transitions.

        Transitions into NORMAL or POSSIBLE_FALL are not persisted (a possible
        fall is not yet a real event). The first persisted state is
        CONFIRMED_FALL, after which RECOVERING/RESOLVED update the same row.
        """
        if t.to_state in (FallState.NORMAL, FallState.POSSIBLE_FALL):
            return
        ev = self.sm.current_event
        if ev is None:
            return
        existing = self.repos.events.get_by_uid(ev.event_uid)
        if existing is None:
            if t.to_state != FallState.CONFIRMED_FALL:
                # We only ever create at confirmation; ignore anything else.
                return
            self.repos.events.create(
                EventRow(
                    event_uid=ev.event_uid,
                    event_type=self.SYNTHETIC_EVENT_TYPE if self.synthetic else ev.event_type,
                    state=ev.state.value,
                    start_time=isoformat(ev.start_time),
                    confirmed_time=isoformat(ev.confirmed_time),
                    end_time=isoformat(ev.end_time),
                    resolved_time=isoformat(ev.resolved_time),
                    highest_confidence=round(ev.highest_confidence, 4),
                    detection_count=ev.detection_count,
                    source_device=ev.source_device,
                )
            )
            # One vitals snapshot per incident, at first confirmation only. Never
            # let a snapshot failure break observe/persistence (also isolated in
            # the callback itself).
            if self._snapshot_fn is not None:
                try:
                    self._snapshot_fn(ev)
                except Exception as exc:  # pragma: no cover - defensive double guard
                    log.warning("incident snapshot hook failed for %s: %s", ev.event_uid, type(exc).__name__)
        else:
            self.repos.events.update(
                ev.event_uid,
                state=ev.state.value,
                confirmed_time=isoformat(ev.confirmed_time),
                end_time=isoformat(ev.end_time),
                resolved_time=isoformat(ev.resolved_time),
                highest_confidence=round(ev.highest_confidence, 4),
                detection_count=ev.detection_count,
            )

    async def _dispatch_alert(self, t: Transition) -> None:
        ev = self.sm.current_event
        if ev is None or self.dispatcher is None:
            if self.dispatcher is None:
                log.warning("Alert requested for %s but no dispatcher configured", t.event_uid)
            # Nothing delivered → do not arm the cooldown.
            self.sm.cancel_pending_alert()
            return
        alert = AlertEvent(
            event_uid=ev.event_uid,
            timestamp=ev.confirmed_time or t.timestamp,
            confidence=ev.highest_confidence,
            source_device=ev.source_device,
            state=ev.state.value,
            detection_count=ev.detection_count,
            simulated=self.simulated,
        )
        try:
            self.last_alert_results = await self.dispatcher.dispatch(alert)
        except Exception as exc:  # pragma: no cover - dispatcher isolates already
            # Defense in depth: never let an alert failure crash observation.
            log.error("Alert dispatch raised unexpectedly for %s: %s", ev.event_uid, exc)
            self.last_alert_results = []
        # Arm the alert cooldown ONLY if at least one provider actually delivered.
        # A failed delivery must not suppress the alert for the next real fall.
        if any(getattr(r, "success", False) for r in self.last_alert_results):
            self.sm.commit_alert()
        else:
            self.sm.cancel_pending_alert()
            log.warning(
                "Alert for %s not delivered by any provider; cooldown not armed", ev.event_uid
            )

    # -- caregiver operations ---------------------------------------------
    async def label_event(self, event_uid: str, label: str) -> EventRow:
        """Apply a human label to an event (real_fall / false_alert / unsure)."""
        valid = {m.value for m in HumanLabel}
        if label not in valid:
            raise ValueError(f"Invalid label {label!r}; expected one of {sorted(valid)}")
        async with self._lock:
            row = self.repos.events.update(event_uid, human_label=label)
            log.info("Event %s labeled %s", event_uid, label)
            return row

    async def resolve_event(self, event_uid: str, note: str | None = None) -> EventRow:
        """Manually resolve an event. If it is the live event, also advance the
        state machine to RESOLVED."""
        async with self._lock:
            existing = self.repos.events.require(event_uid)
            now = self.clock.now()  # single clock read for consistent timestamps
            live = self.sm.current_event
            fields: dict[str, Any] = {"resolution_note": note} if note is not None else {}
            # Only CONFIRMED/RECOVERING events are both live AND persisted; a
            # POSSIBLE event has no DB row (require() above would have raised),
            # so it is not a resolvable live state here.
            is_live = (
                live is not None
                and live.event_uid == event_uid
                and self.sm.state in (FallState.CONFIRMED_FALL, FallState.RECOVERING)
            )
            if is_live:
                # The state machine sets resolved_time/end_time; _persist_transition
                # writes them. We only add the caregiver note here (no duplicate
                # state/time writes, no second clock read).
                for t in self.sm.manual_resolve(now=now):
                    self._persist_transition(t)
            elif existing.state != FallState.RESOLVED.value:
                fields["state"] = FallState.RESOLVED.value
                fields["resolved_time"] = existing.resolved_time or isoformat(now)
                if existing.end_time is None:
                    fields["end_time"] = isoformat(now)
            row = self.repos.events.update(event_uid, **fields) if fields else self.repos.events.require(event_uid)
            log.info("Event %s resolved (note=%s)", event_uid, bool(note))
            return row

    async def reset(self) -> list[Transition]:
        async with self._lock:
            transitions = self.sm.reset()
            log.info("Event state machine reset")
            return transitions

    # -- status ------------------------------------------------------------
    def current_state(self) -> FallState:
        return self.sm.state

    def status(self) -> dict[str, Any]:
        snap = self.sm.snapshot()
        return {
            "fall_state": snap["state"],
            "active_event_uid": snap["event_uid"],
            "highest_confidence": snap["highest_confidence"],
            "detection_count": snap["detection_count"],
        }
