"""Posture transition gate — a clean reproduction of the legacy VytalLink v1
"DTS-lite" false-positive protection.

The legacy live detector (`vytalinkv1/src/test_fall_fast.py`) did **not** treat
every ``fallen`` detection as a fall. It only raised a *fall event* when a
``fallen`` posture **followed a recent upright → fallen transition** — i.e. it
distinguished a real fall from someone who was **already lying down** or who
**slowly lies down**. That gate (`min_upright_frames`/`min_fallen_frames`, a
bounded transition window, and an anti-stale history reset) is reproduced here
in a small, deterministic, clock-injected form so it is fully unit-testable with
``ManualClock`` and so the new ``YoloFallDetector`` can surface only genuine
falls as fall *evidence* to the existing state machine.

This is intentionally simpler than the legacy velocity/aspect-ratio scoring: the
heavy temporal *confirmation* now lives in the state machine
(``FALL_CONFIRM_SECONDS``/``FALL_CLEAR_SECONDS``). The gate's single job is to
decide, per frame, whether the current ``fallen`` posture is part of a real
fall transition. It is optional (``DETECTOR_REQUIRE_TRANSITION``) and degrades
to "any fallen posture is evidence" when disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vytallink.common.clock import Clock, SystemClock
from vytallink.common.logging_setup import get_logger

log = get_logger("vision.posture_gate")


@dataclass(slots=True)
class GateConfig:
    #: Upright (sitting/standing) observations that establish "was upright".
    #: Counted with gap tolerance — sparse detections still accumulate.
    min_upright_frames: int = 2
    #: Consecutive fallen detections required to confirm the transition. No-detection
    #: gaps do NOT break the run (missing data ≠ "stood up"); a clear upright does.
    min_fallen_frames: int = 2
    #: Confidence at/above which a sit/stand detection counts as "upright".
    upright_conf: float = 0.50
    #: Confidence at/above which a class-0 detection counts as "fallen".
    fallen_conf: float = 0.60
    #: The fallen run must begin within this many seconds of having been upright,
    #: otherwise it is treated as "already lying" / too-slow.
    transition_window_seconds: float = 2.5
    #: Clear the upright/fallen memory after this long with no detection at all.
    stale_seconds: float = 2.0


class PostureTransitionGate:
    """Decide whether the current ``fallen`` posture is a genuine fall event.

    Feed one observation per processed frame via :meth:`observe`. Returns
    ``True`` only on a ``fallen`` frame that is part of a confirmed upright→fallen
    transition; the result latches across sustained fallen frames and clears when
    the subject becomes upright again or detection goes stale.
    """

    def __init__(self, config: GateConfig | None = None, *, clock: Clock | None = None) -> None:
        self.cfg = config or GateConfig()
        self.clock: Clock = clock or SystemClock()
        self.reset()

    def reset(self) -> None:
        self._upright_count = 0
        self._fallen_run = 0
        self._seen_upright_mono: float | None = None
        self._last_detect_mono: float | None = None
        self._fall_active = False

    @property
    def fall_active(self) -> bool:
        return self._fall_active

    def observe(
        self,
        *,
        fallen_conf: float,
        upright_conf: float,
        has_detection: bool,
        mono: float | None = None,
    ) -> bool:
        """Update the gate with one frame's best fallen/upright confidences.

        Args:
            fallen_conf: best class-0 (``fallen``) confidence this frame (0 if none).
            upright_conf: best sit/stand confidence this frame (0 if none).
            has_detection: whether any person box was detected this frame.
            mono: optional monotonic timestamp (defaults to the injected clock).

        Returns:
            ``True`` iff this frame is genuine fall evidence.
        """
        mono = self.clock.monotonic() if mono is None else mono

        # Anti-stale: a gap with no detection at all wipes the transition memory
        # so a person re-entering the frame already on the floor is not a "fall".
        if (
            self._last_detect_mono is not None
            and (mono - self._last_detect_mono) > self.cfg.stale_seconds
        ):
            self.reset()

        if has_detection:
            self._last_detect_mono = mono

        is_fallen = has_detection and fallen_conf >= self.cfg.fallen_conf
        is_upright = has_detection and upright_conf >= self.cfg.upright_conf and not is_fallen

        if is_upright:
            # Gap-tolerant upright accumulation: real footage detects an upright
            # person only intermittently, so we count cumulatively (reset only by a
            # fall or a stale gap) and remember the most recent upright time.
            self._upright_count += 1
            self._fallen_run = 0
            if self._upright_count >= self.cfg.min_upright_frames:
                self._seen_upright_mono = mono
            self._fall_active = False  # they are upright now
            return False

        if is_fallen:
            # No-detection gaps do NOT break the fallen run (missing data, not a
            # "stood up" signal); only a clear upright frame resets it (above).
            self._fallen_run += 1
            self._upright_count = 0
            if not self._fall_active and self._fallen_run >= self.cfg.min_fallen_frames:
                up = self._seen_upright_mono
                if up is not None and (mono - up) <= self.cfg.transition_window_seconds:
                    self._fall_active = True
                    log.info("Posture gate: upright→fallen transition confirmed")
            return self._fall_active

        # Ambiguous frame (no detection, or detection but neither clearly upright
        # nor fallen): preserve all run/latch state. A detection gap during a
        # confirmed fall is bridged here; the state machine's recovery window
        # handles longer gaps. No *new* evidence is reported this frame.
        return False

    def debug_state(self) -> dict[str, Any]:
        return {
            "upright_count": self._upright_count,
            "fallen_run": self._fallen_run,
            "fall_active": self._fall_active,
            "has_upright_memory": self._seen_upright_mono is not None,
        }
