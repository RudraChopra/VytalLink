"""Fall event states, labels, and the transition record type."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FallState(str, Enum):
    """Lifecycle states for a fall event.

    * ``NORMAL``         — no fall evidence; monitoring.
    * ``POSSIBLE_FALL``  — evidence seen, accumulating toward confirmation.
    * ``CONFIRMED_FALL`` — evidence sustained past the confirm window; alerted.
    * ``RECOVERING``     — evidence cleared after a confirmed fall; awaiting the
                            clear window before resolving.
    * ``RESOLVED``       — event finished (recovered or manually resolved).
    """

    NORMAL = "normal"
    POSSIBLE_FALL = "possible_fall"
    CONFIRMED_FALL = "confirmed_fall"
    RECOVERING = "recovering"
    RESOLVED = "resolved"


#: States in which an event is "active" and persisted/updated in the database.
ACTIVE_STATES = frozenset(
    {FallState.CONFIRMED_FALL, FallState.RECOVERING, FallState.RESOLVED}
)


class HumanLabel(str, Enum):
    """Caregiver-applied label for an event."""

    REAL_FALL = "real_fall"
    FALSE_ALERT = "false_alert"
    UNSURE = "unsure"


class TransitionReason(str, Enum):
    EVIDENCE_DETECTED = "evidence_detected"
    SUSTAINED_EVIDENCE = "sustained_evidence"
    EVIDENCE_CLEARED = "evidence_cleared"
    EVIDENCE_RETURNED = "evidence_returned"
    BRIEF_EVIDENCE_DISMISSED = "brief_evidence_dismissed"
    RECOVERY_TIMEOUT = "recovery_timeout"
    NEW_EVENT_AFTER_RESOLVED = "new_event_after_resolved"
    READY_AFTER_RESOLVED = "ready_after_resolved"
    MANUAL_RESOLVE = "manual_resolve"
    RESET = "reset"


@dataclass(slots=True)
class Transition:
    """A single state transition emitted by the state machine."""

    timestamp: datetime
    from_state: FallState
    to_state: FallState
    event_uid: str
    reason: TransitionReason
    alert: bool = False
    highest_confidence: float = 0.0
    detection_count: int = 0

    def describe(self) -> str:
        return (
            f"{self.from_state.value} -> {self.to_state.value} "
            f"({self.reason.value}) event={self.event_uid} "
            f"conf={self.highest_confidence:.2f} alert={self.alert}"
        )
