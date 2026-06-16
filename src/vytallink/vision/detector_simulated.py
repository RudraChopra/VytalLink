"""Deterministic simulated fall detector.

A *real* working provider labeled "simulation". It emits raw detections driven
by a controllable scenario:

* ``NORMAL`` — a benign ``person`` (standing) detection, i.e. no fall evidence.
* ``FALL``   — a ``fall`` detection above threshold (fall evidence).
* a scripted sequence of booleans for fine-grained tests.

Confidence varies deterministically with frame id (no RNG) so output is
realistic yet reproducible. The simulation driver flips the scenario in
response to the dashboard / API simulation controls.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any

from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, RawDetection
from vytallink.vision.detector_base import FallDetector

log = get_logger("vision.detector.simulated")


class Scenario(str, Enum):
    NORMAL = "normal"
    FALL = "fall"


class SimulatedFallDetector(FallDetector):
    name = "simulated"

    #: Class ids/names mirror a plausible 2-class fall model.
    FALL_CLASS_ID = 1
    FALL_CLASS_NAME = "fall"
    PERSON_CLASS_ID = 0
    PERSON_CLASS_NAME = "person"

    def __init__(
        self,
        *,
        scenario: Scenario = Scenario.NORMAL,
        fall_confidence: float = 0.9,
        person_confidence: float = 0.8,
    ) -> None:
        self._scenario = scenario
        self.fall_confidence = fall_confidence
        self.person_confidence = person_confidence
        self._script: deque[bool] = deque()
        self._loaded = False

    # -- control -----------------------------------------------------------
    def load(self) -> None:
        self._loaded = True
        log.info("Simulated fall detector loaded (scenario=%s)", self._scenario.value)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def set_scenario(self, scenario: Scenario | str) -> None:
        self._scenario = Scenario(scenario)
        self._script.clear()
        log.debug("Simulated detector scenario -> %s", self._scenario.value)

    def set_script(self, evidence_sequence: list[bool]) -> None:
        """Provide an explicit per-call evidence sequence (test helper)."""
        self._script = deque(evidence_sequence)

    # -- inference ---------------------------------------------------------
    def infer(self, frame: Frame) -> list[RawDetection]:
        emit_fall: bool
        if self._script:
            emit_fall = bool(self._script.popleft())
        else:
            emit_fall = self._scenario == Scenario.FALL

        # Deterministic confidence variation in [base, base+0.09].
        jitter = (frame.frame_id % 10) * 0.01
        if emit_fall:
            conf = min(0.99, self.fall_confidence + jitter)
            return [
                RawDetection(
                    timestamp=frame.timestamp,
                    class_id=self.FALL_CLASS_ID,
                    class_name=self.FALL_CLASS_NAME,
                    confidence=round(conf, 3),
                    bbox=(0.30, 0.55, 0.70, 0.95),
                    source_id=frame.source_id,
                    frame_id=frame.frame_id,
                    metadata={"simulated": True, "scenario": self._scenario.value},
                )
            ]
        conf = min(0.99, self.person_confidence + jitter)
        return [
            RawDetection(
                timestamp=frame.timestamp,
                class_id=self.PERSON_CLASS_ID,
                class_name=self.PERSON_CLASS_NAME,
                confidence=round(conf, 3),
                bbox=(0.40, 0.20, 0.60, 0.90),
                source_id=frame.source_id,
                frame_id=frame.frame_id,
                metadata={"simulated": True, "scenario": self._scenario.value},
            )
        ]

    def health(self) -> dict[str, Any]:
        h = super().health()
        h["scenario"] = self._scenario.value
        h["simulated"] = True
        return h
