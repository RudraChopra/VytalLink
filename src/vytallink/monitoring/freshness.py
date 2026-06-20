"""Freshness classification for vitals and camera frames.

Pure and deterministic: classification is by AGE against operator-tunable
thresholds. Old data is never treated as current — a missing sample is
``unavailable``, never ``fresh``. Thresholds are operational, NOT medical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# vitals freshness states
VITALS_FRESH = "fresh"
VITALS_AGING = "aging"
VITALS_STALE = "stale"
VITALS_UNAVAILABLE = "unavailable"

# camera freshness states
CAM_FRESH = "fresh"
CAM_STALE = "stale"
CAM_OFFLINE = "offline"
CAM_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FreshnessThresholds:
    vitals_fresh: float = 15.0
    vitals_aging: float = 45.0
    vitals_stale: float = 90.0
    camera_fresh: float = 5.0
    camera_stale: float = 15.0

    @classmethod
    def from_settings(cls, s: Any) -> "FreshnessThresholds":
        return cls(
            vitals_fresh=float(s.vitals_fresh_seconds),
            vitals_aging=float(s.vitals_aging_seconds),
            vitals_stale=float(s.vitals_stale_seconds),
            camera_fresh=float(s.camera_frame_fresh_seconds),
            camera_stale=float(s.camera_frame_stale_seconds),
        )


def vitals_freshness(age_seconds: float | None, t: FreshnessThresholds) -> str:
    """fresh | aging | stale | unavailable from a sample age (None = no data).

    Uses all three thresholds: a sample older than ``vitals_stale`` is treated as
    ``unavailable`` (too old to represent the patient's current state)."""
    if age_seconds is None:
        return VITALS_UNAVAILABLE
    if age_seconds <= t.vitals_fresh:
        return VITALS_FRESH
    if age_seconds <= t.vitals_aging:
        return VITALS_AGING
    if age_seconds <= t.vitals_stale:
        return VITALS_STALE
    return VITALS_UNAVAILABLE  # older than the stale bound -> no current data


def vitals_is_usable(freshness: str) -> bool:
    """Whether a vitals value may be used as positive evidence. Stale/unavailable
    vitals are NOT usable — they must never read as reassuring."""
    return freshness in (VITALS_FRESH, VITALS_AGING)


def camera_freshness(age_seconds: float | None, *, connected: bool, t: FreshnessThresholds) -> str:
    """fresh | stale | offline | unavailable for one camera frame age."""
    if not connected:
        return CAM_OFFLINE
    if age_seconds is None:
        return CAM_UNAVAILABLE
    if age_seconds <= t.camera_fresh:
        return CAM_FRESH
    return CAM_STALE


def camera_is_fresh(freshness: str) -> bool:
    """A camera whose state may drive the aggregate vision state."""
    return freshness == CAM_FRESH
