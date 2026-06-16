"""Fall detector interface and the evidence-mapping helper.

A detector turns a :class:`Frame` into zero or more :class:`RawDetection`
objects. A raw detection is *not* a confirmed fall — the state machine
aggregates detections over time. The :func:`detections_to_evidence` helper maps
a detection list to ``(evidence, confidence)`` using the configured fall-class
names and confidence threshold.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable

from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus, RawDetection

log = get_logger("vision.detector")


class FallDetector(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def load(self) -> None:  # pragma: no cover - interface
        """Load model/weights. Cheap for simulation; heavy for real adapters."""
        ...

    @abc.abstractmethod
    def infer(self, frame: Frame) -> list[RawDetection]:  # pragma: no cover - interface
        """Run detection on one frame."""
        ...

    def close(self) -> None:
        return None

    @property
    def loaded(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {
            "status": (HealthStatus.OK if self.loaded else HealthStatus.DOWN).value,
            "name": self.name,
            "loaded": self.loaded,
        }


def detections_to_evidence(
    detections: Iterable[RawDetection],
    fall_classes: set[str],
    confidence_threshold: float,
) -> tuple[bool, float]:
    """Return ``(is_fall_evidence, max_confidence)`` for the frame.

    Evidence requires at least one detection whose class is a configured
    fall-class AND whose confidence meets the threshold.
    """
    best = 0.0
    evidence = False
    for d in detections:
        if d.class_name.lower() in fall_classes and d.confidence >= confidence_threshold:
            evidence = True
            best = max(best, d.confidence)
    return evidence, best
