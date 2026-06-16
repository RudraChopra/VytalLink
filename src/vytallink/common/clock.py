"""Clock abstraction enabling deterministic, sleep-free time-based tests.

Two notions of time are exposed:

* ``now()``        — a timezone-aware UTC ``datetime`` used for stored
                     timestamps (event start/confirm/resolve, vitals, audit).
* ``monotonic()``  — a float seconds counter used to measure *durations*
                     (confirmation window, recovery window, alert cooldown).
                     A monotonic source never goes backwards on wall-clock
                     adjustments, which is exactly what we want for timing.

``SystemClock`` is used in production. ``ManualClock`` is used in tests and by
the simulation driver so timing logic can be exercised instantly and
deterministically without real sleeps.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source abstraction."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic seconds counter for measuring durations."""
        ...


class SystemClock:
    """Real wall-clock + monotonic clock. Used in production."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


class ManualClock:
    """Deterministic clock for tests and the simulation driver.

    ``now()`` and ``monotonic()`` advance together via :meth:`advance`, so a
    caller can simulate "N seconds passing" without sleeping.
    """

    def __init__(self, start: datetime | None = None, monotonic_start: float = 1000.0) -> None:
        if start is None:
            start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        self._now = start.astimezone(timezone.utc)
        self._mono = float(monotonic_start)

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        """Move both time sources forward by ``seconds``."""
        if seconds < 0:
            raise ValueError("Cannot advance a clock by a negative amount")
        self._mono += seconds
        self._now = self._now + timedelta(seconds=seconds)

    def set_now(self, when: datetime) -> None:
        """Anchor wall-clock time without changing the monotonic counter."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        self._now = when.astimezone(timezone.utc)


def utcnow() -> datetime:
    """Convenience: current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def isoformat(dt: datetime | None) -> str | None:
    """Serialize a datetime to an ISO-8601 string with a ``Z``-style offset.

    Returns ``None`` for ``None`` so it round-trips cleanly through JSON.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
