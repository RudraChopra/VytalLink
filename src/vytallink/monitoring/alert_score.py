"""Explainable multimodal alert scoring.

Pure and deterministic. Combines the aggregated vision state with the latest
*usable* vitals into a small {level, score, reasons[]} view. It is INFORMATIONAL
ONLY — it is exposed in the API/patient-state and is NEVER wired to the alert
dispatcher (the only thing that dispatches is the fall-event pipeline). It is not
a medical diagnosis. Thresholds are operator-tunable.

Principles:
  * A confirmed fall is high priority even if vitals look normal.
  * Abnormal vitals without a fall are handled independently.
  * Stale/unavailable vitals are never treated as reassuring — they are flagged,
    and out-of-range checks only run on usable (fresh/aging) vitals.
  * One offline camera never erases another camera's evidence (the caller passes
    the worst-of-FRESH-cameras vision state).
  * Confidence values from different cameras are never summed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# severity levels (low -> high) and their numeric score
LEVEL_NORMAL = "normal"
LEVEL_INFO = "info"
LEVEL_WARNING = "warning"
LEVEL_CRITICAL = "critical"
_LEVEL_RANK = {LEVEL_NORMAL: 0, LEVEL_INFO: 1, LEVEL_WARNING: 2, LEVEL_CRITICAL: 3}

# reason codes
R_FALL_CONFIRMED = "fall_confirmed"
R_FALL_SUSPECTED = "fall_suspected"
R_HR_HIGH = "heart_rate_high"
R_HR_LOW = "heart_rate_low"
R_RR_HIGH = "respiratory_rate_high"
R_RR_LOW = "respiratory_rate_low"
R_VITALS_STALE = "vitals_stale"
R_VITALS_UNAVAILABLE = "vitals_unavailable"
R_VISION_UNAVAILABLE = "vision_unavailable"
R_PERSON_AMBIGUOUS = "person_count_ambiguous"
R_INCIDENT_ACTIVE = "incident_active"


@dataclass(frozen=True)
class ScoreThresholds:
    hr_low: float = 40.0
    hr_high: float = 120.0
    rr_low: float = 8.0
    rr_high: float = 30.0

    @classmethod
    def from_settings(cls, s: Any) -> "ScoreThresholds":
        return cls(hr_low=float(s.vitals_hr_low), hr_high=float(s.vitals_hr_high),
                   rr_low=float(s.vitals_rr_low), rr_high=float(s.vitals_rr_high))


def score_patient(
    *,
    vision_overall_state: str,
    vision_available: bool,
    any_camera_fresh: bool,
    person_ambiguous: bool,
    incident_active: bool,
    heart_rate: float | None,
    respiratory_rate: float | None,
    vitals_usable: bool,
    vitals_freshness: str,
    thr: ScoreThresholds,
) -> dict[str, Any]:
    """Return {"level", "score", "reasons": [code,...]}. NOT a diagnosis."""
    reasons: list[tuple[str, str]] = []  # (code, level)

    # --- vision (fall) — the caller passes the worst-of-FRESH-cameras state ---
    if vision_overall_state == "confirmed_fall":
        reasons.append((R_FALL_CONFIRMED, LEVEL_CRITICAL))
    elif vision_overall_state in ("possible_fall", "recovering"):
        reasons.append((R_FALL_SUSPECTED, LEVEL_WARNING))

    if not vision_available or not any_camera_fresh:
        # No camera at all, or cameras present but none fresh -> we cannot see.
        reasons.append((R_VISION_UNAVAILABLE, LEVEL_WARNING))
    if person_ambiguous:
        reasons.append((R_PERSON_AMBIGUOUS, LEVEL_INFO))

    # --- vitals — only USABLE vitals are evaluated for abnormality ---
    if vitals_usable:
        if heart_rate is not None:
            if heart_rate > thr.hr_high:
                reasons.append((R_HR_HIGH, LEVEL_WARNING))
            elif heart_rate < thr.hr_low:
                reasons.append((R_HR_LOW, LEVEL_WARNING))
        if respiratory_rate is not None:
            if respiratory_rate > thr.rr_high:
                reasons.append((R_RR_HIGH, LEVEL_WARNING))
            elif respiratory_rate < thr.rr_low:
                reasons.append((R_RR_LOW, LEVEL_WARNING))
    elif vitals_freshness == "unavailable":
        reasons.append((R_VITALS_UNAVAILABLE, LEVEL_INFO))
    else:  # stale: present but not usable — flagged, never reassuring
        reasons.append((R_VITALS_STALE, LEVEL_INFO))

    if incident_active:
        reasons.append((R_INCIDENT_ACTIVE, LEVEL_WARNING))

    level = LEVEL_NORMAL
    for _code, lvl in reasons:
        if _LEVEL_RANK[lvl] > _LEVEL_RANK[level]:
            level = lvl
    return {"level": level, "score": _LEVEL_RANK[level], "reasons": [c for c, _ in reasons]}
