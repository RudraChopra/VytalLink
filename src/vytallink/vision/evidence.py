"""Fall-evidence smoother — bridges brief real-world detection gaps.

Real YOLO detection on a live stream is intermittent: a genuinely fallen person
is detected on most frames but missed on some (occlusion, motion blur, angle).
The fall-event state machine treats a single no-evidence frame as "evidence
gone", so raw per-frame evidence would keep resetting a real fall before it can
confirm. The legacy v1 detector avoided this with a 5 s rolling history; we
reproduce just that gap-bridging here, cleanly and testably, **without touching
the state machine**.

Behaviour (clock-injected, deterministic):

* a ``fallen`` frame → evidence ``True`` and (re)arms a short hold;
* a no-detection / ambiguous frame within the hold window → evidence stays
  ``True`` (bridge the gap), carrying the last confidence;
* a frame with a clear **upright** (non-fall) detection → cancels the hold
  immediately (the person is up) → evidence ``False``;
* once the hold expires with no new fallen frame → evidence ``False``.

The hold is shorter than ``FALL_CLEAR_SECONDS`` so genuine recovery still works.
Used only in live mode; simulation is unaffected.
"""

from __future__ import annotations

from vytallink.common.clock import Clock, SystemClock


class FallEvidenceSmoother:
    def __init__(self, hold_seconds: float = 1.0, *, clock: Clock | None = None) -> None:
        self.hold_seconds = float(hold_seconds)
        self.clock: Clock = clock or SystemClock()
        self._hold_until: float | None = None
        self._last_conf = 0.0

    def reset(self) -> None:
        self._hold_until = None
        self._last_conf = 0.0

    @property
    def holding(self) -> bool:
        return self._hold_until is not None

    def update(
        self,
        raw_evidence: bool,
        confidence: float,
        *,
        had_detection: bool,
        had_upright: bool,
        mono: float | None = None,
    ) -> tuple[bool, float]:
        """Return the smoothed ``(evidence, confidence)`` for this frame."""
        mono = self.clock.monotonic() if mono is None else mono

        if raw_evidence:
            self._hold_until = mono + self.hold_seconds
            self._last_conf = confidence
            return True, confidence

        # A clear upright detection means the person is NOT down: end the hold now.
        if had_upright:
            self._hold_until = None
            return False, 0.0

        # No fall and no clear upright (a detection gap): bridge it while held.
        if self._hold_until is not None and mono < self._hold_until:
            return True, self._last_conf

        self._hold_until = None
        return False, 0.0
