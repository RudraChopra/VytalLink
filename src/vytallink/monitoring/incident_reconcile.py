"""Stale-incident reconciliation policy (pure, deterministic).

Prevents an old unresolved incident from pinning the patient alert baseline
forever, WITHOUT closing a genuinely ongoing fall. The hard part: a sustained
``confirmed_fall`` does not advance ``updated_at`` (no state transition), so age
alone would wrongly close it — we therefore require the source camera's CURRENT
live state to decide.

Resolve only on POSITIVE evidence:
  * the source camera is orphaned (not in the current config — no camera can ever
    support the incident again), or
  * the source camera is present, fresh, and currently NORMAL (recovered).

A configured-but-offline/stale source camera is AMBIGUOUS (absence of evidence,
not evidence of recovery): keep the incident open and mark the system degraded.
"""

from __future__ import annotations

#: Fall states that mean the source camera currently SUPPORTS an active incident.
ACTIVE_FALL_STATES = frozenset({"confirmed_fall", "possible_fall", "recovering"})

# resolution reasons
REASON_STALE_TIMEOUT = "stale_incident_timeout"
REASON_CAMERA_RECOVERED = "camera_recovered"


def classify_incident(
    *,
    age_seconds: float,
    stale_seconds: float,
    source_in_config: bool,
    camera_fresh: bool,
    camera_fall_state: str | None,
) -> tuple[bool, str | None, bool]:
    """Return ``(resolve, reason, ambiguous)`` for one unresolved incident."""
    if age_seconds <= stale_seconds:
        return (False, None, False)                       # recent — guards brief frame delays
    if not source_in_config:
        return (True, REASON_STALE_TIMEOUT, False)        # orphaned — no camera can support it
    if camera_fresh and camera_fall_state in ACTIVE_FALL_STATES:
        return (False, None, False)                       # genuinely ongoing, supported by camera
    if camera_fresh and camera_fall_state == "normal":
        return (True, REASON_CAMERA_RECOVERED, False)     # positive recovery evidence
    return (False, None, True)                            # configured but offline/stale -> ambiguous
