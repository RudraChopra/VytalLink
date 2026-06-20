"""Simultaneous multi-camera fall detection over RTSP.

Design (Apple-silicon / MPS constraints in mind):

* **One** fall-detection model is loaded **once** and shared by every camera.
* **One** single-thread inference lane serializes all inference. MPS/Metal
  command buffers are not safe across threads, so every ``infer()`` runs on the
  same dedicated thread; this also makes scheduling *fair* (FIFO over the two
  cameras) and bounds the inference backlog to at most one in-flight frame per
  camera (no unbounded queue).
* Each camera runs its **own** capture loop in its own thread with its own
  camera id, frame timestamps, dropped/reconnect counters, evidence smoother,
  fall-confirmation state machine, event history, metrics, and health. A failure
  in one camera's loop is caught and isolated — it can neither crash nor block
  the other camera.

Nothing here saves video/images, and no credential or RTSP URL is ever logged
or surfaced (health uses the credential-free camera id only).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from vytallink.common.clock import Clock, SystemClock, isoformat
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, RawDetection
from vytallink.config.cameras import CameraConfig
from vytallink.events.state_machine import FallEventStateMachine
from vytallink.events.states import FallState, Transition
from vytallink.vision.detector_base import FallDetector, detections_to_evidence
from vytallink.vision.evidence import FallEvidenceSmoother

log = get_logger("vision.multi_camera")

#: Type of the per-camera observe bridge: evidence + confidence -> transitions.
ObserveFn = Callable[[bool, float], list[Transition]]
#: Type of the shared serialized inference call.
InferFn = Callable[[Frame], list[RawDetection]]


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[k], 2)


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


class CameraWorker:
    """One camera's capture→inference→state-machine loop, isolated in a thread."""

    def __init__(
        self,
        config: CameraConfig,
        camera: Any,                 # CameraProvider (RTSPCamera in production)
        state_machine: FallEventStateMachine,
        observe_fn: ObserveFn,
        infer_fn: InferFn,
        *,
        fall_class_set: set[str],
        confidence_threshold: float,
        clock: Clock | None = None,
        evidence_smoother: FallEvidenceSmoother | None = None,
        max_fps: float = 12.0,
    ) -> None:
        self.config = config
        self.camera_id = config.camera_id
        self.camera = camera
        self.sm = state_machine
        self._observe = observe_fn
        self._infer = infer_fn
        self.fall_class_set = fall_class_set
        self.confidence_threshold = confidence_threshold
        self.clock: Clock = clock or SystemClock()
        self._smoother = evidence_smoother
        self._min_interval = 1.0 / max(0.1, max_fps)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()  # guards the mutable metric collections

        # metrics (primitive counters read locklessly; collections under _lock)
        self._frames_received = 0
        self._failed_reads = 0
        self._processed = 0
        self._confirmed_falls = 0
        self._tick_errors = 0
        self._inflight = 0          # frames this worker has in the inference lane (0/1)
        self._last_frame_id: int | None = None
        self._read_ms: deque[float] = deque(maxlen=512)
        self._infer_ms: deque[float] = deque(maxlen=512)
        self._class_counts: Counter[str] = Counter()
        self._events: list[dict[str, Any]] = []   # confirmed-fall event history

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        try:
            self.camera.open()
        except Exception as exc:  # camera will retry with its own backoff
            log.warning("camera %s did not open at start (%s); will retry", self.camera_id, type(exc).__name__)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"vyt-cam-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the loop to stop and join. Returns True if it stopped cleanly."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
            alive = t.is_alive()
        else:
            alive = False
        try:
            self.camera.close()
        except Exception:  # pragma: no cover - defensive
            pass
        return not alive

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- loop --------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            t_loop = self.clock.monotonic()
            try:
                self._tick()
            except Exception as exc:  # ISOLATION: never let one camera crash the worker/others
                self._tick_errors += 1
                log.warning("camera %s tick error: %s", self.camera_id, type(exc).__name__)
            dt = self.clock.monotonic() - t_loop
            self._stop.wait(max(0.0, self._min_interval - dt))

    def _tick(self) -> None:
        t0 = time.perf_counter()
        frame = self.camera.read()
        read_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._read_ms.append(read_ms)
        if frame is None:
            self._failed_reads += 1
            return
        self._frames_received += 1
        if frame.image is None:
            return

        # Shared, serialized inference (fair FIFO over cameras; bounded backlog).
        # A single inference failure is isolated: record it and skip this frame
        # rather than aborting the loop (and never affect the other camera).
        self._inflight = 1
        ti = time.perf_counter()
        try:
            detections = self._infer(frame)
        except Exception as exc:
            self._tick_errors += 1
            log.warning("camera %s inference error: %s", self.camera_id, type(exc).__name__)
            return
        finally:
            self._inflight = 0
        infer_ms = (time.perf_counter() - ti) * 1000.0
        with self._lock:
            self._infer_ms.append(infer_ms)
            for d in detections:
                self._class_counts[d.class_name] += 1
        self._processed += 1

        evidence, confidence = detections_to_evidence(
            detections, self.fall_class_set, self.confidence_threshold
        )
        if self._smoother is not None:
            had_detection = bool(detections)
            had_upright = any(d.class_name.lower() not in self.fall_class_set for d in detections)
            evidence, confidence = self._smoother.update(
                evidence, confidence, had_detection=had_detection, had_upright=had_upright
            )

        for t in self._observe(evidence, confidence):
            if t.to_state == FallState.CONFIRMED_FALL:
                self._confirmed_falls += 1
                with self._lock:
                    self._events.append(
                        {"event_uid": t.event_uid, "time": isoformat(t.timestamp), "confidence": round(confidence, 3)}
                    )

    # -- health / metrics --------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Credential-free per-camera health. No host/URL/username is exposed."""
        ch = self.camera.health()
        age_s = ch.get("last_frame_age_seconds")
        with self._lock:
            read = list(self._read_ms)
            infer = list(self._infer_ms)
            classes = dict(self._class_counts)
        return {
            "connected": bool(ch.get("opened")),
            "status": ch.get("status"),
            "fps": ch.get("effective_fps", 0.0),                 # unique capture fps
            "last_frame_age_ms": round(age_s * 1000.0, 1) if age_s is not None else None,
            "reconnects": ch.get("reconnects", 0),
            "resolution": ch.get("resolution"),
            "frames_received": self._frames_received,
            "failed_reads": self._failed_reads,
            "frames_processed": self._processed,
            "dropped_frames": ch.get("frames_dropped"),
            "stale": bool(ch.get("stale")),
            "backlog": self._inflight,                            # frames awaiting inference (0/1)
            "fall_state": self.sm.state.value,
            "confirmed_falls": self._confirmed_falls,
            "detected_classes": classes,
            "read_ms_avg": _avg(read),
            "read_ms_p95": _pct(read, 95),
            "infer_ms_avg": _avg(infer),
            "infer_ms_p95": _pct(infer, 95),
            "tick_errors": self._tick_errors,
            "alive": self.alive,
        }

    def metrics(self, elapsed: float) -> dict[str, Any]:
        """Diagnostic metrics (adds throughput rates over ``elapsed`` seconds)."""
        h = self.health()
        h["inference_fps"] = round(self._processed / elapsed, 2) if elapsed > 0 else 0.0
        # End-to-end = frames that completed the full capture→infer→observe path.
        h["end_to_end_fps"] = h["inference_fps"]
        h["events"] = list(self._events)
        return h


class MultiCameraMonitor:
    """Owns one shared model + one inference lane + N isolated camera workers."""

    def __init__(
        self,
        detector: FallDetector,
        workers_factory: Callable[["MultiCameraMonitor"], list[CameraWorker]],
        *,
        clock: Clock | None = None,
    ) -> None:
        self.detector = detector
        self.clock: Clock = clock or SystemClock()
        self._infer_executor: ThreadPoolExecutor | None = None
        self._model_load_count = 0
        self._started_at: float | None = None
        self._running = False

        # inference-lane backlog accounting (queue depth across all cameras)
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._peak_pending = 0

        self._workers: list[CameraWorker] = workers_factory(self)

    # -- shared serialized inference (the single fair lane) ----------------
    def infer(self, frame: Frame) -> list[RawDetection]:
        """Submit inference to the single dedicated thread and block for the
        result. Serialized (MPS-safe) and FIFO-fair across cameras."""
        executor = self._infer_executor
        if executor is None:  # pragma: no cover - start() sets this
            return self.detector.infer(frame)
        with self._pending_lock:
            self._pending += 1
            self._peak_pending = max(self._peak_pending, self._pending)
        try:
            return executor.submit(self.detector.infer, frame).result()
        finally:
            with self._pending_lock:
                self._pending -= 1

    @property
    def workers(self) -> list[CameraWorker]:
        return self._workers

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vyt-multi-infer")
        # Load + warm the model ONCE, on the dedicated inference thread.
        self._infer_executor.submit(self.detector.load).result()
        self._model_load_count += 1
        self._started_at = self.clock.monotonic()
        for w in self._workers:
            w.start()
        self._running = True
        log.info("MultiCameraMonitor started with %d camera(s); model loaded once", len(self._workers))

    def stop(self) -> dict[str, bool]:
        """Stop every worker and the inference lane. Returns per-camera clean-stop."""
        results: dict[str, bool] = {}
        for w in self._workers:
            results[w.camera_id] = w.stop()
        if self._infer_executor is not None:
            self._infer_executor.shutdown(wait=True)
            self._infer_executor = None
        self._running = False
        try:
            self.detector.close()
        except Exception:  # pragma: no cover - defensive
            pass
        log.info("MultiCameraMonitor stopped (clean=%s)", results)
        return results

    @property
    def elapsed(self) -> float:
        return (self.clock.monotonic() - self._started_at) if self._started_at else 0.0

    @property
    def queue_depth(self) -> int:
        return self._pending

    @property
    def peak_queue_depth(self) -> int:
        return self._peak_pending

    def all_workers_alive(self) -> bool:
        return all(w.alive for w in self._workers)

    # -- health ------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """The ``vision`` health block: mode + per-camera (credential-free)."""
        return {
            "mode": "rtsp_multi",
            "model_load_count": self._model_load_count,
            "inference_queue_depth": self._pending,
            "inference_queue_peak": self._peak_pending,
            "cameras": {w.camera_id: w.health() for w in self._workers},
        }


def make_event_bridge(
    event_manager: Any,
    loop: "asyncio.AbstractEventLoop",
    *,
    camera_id: str,
    timeout: float = 10.0,
) -> ObserveFn:
    """Bridge a synchronous :class:`CameraWorker` to an async ``EventManager``.

    Each camera worker runs in its own thread, but ``EventManager.observe`` is a
    coroutine that persists events and dispatches alerts and MUST run on the
    app's event loop. This returns a synchronous ``observe(evidence, confidence)``
    that schedules the coroutine onto ``loop`` and blocks the worker thread for
    the result, so the worker still receives its list of transitions.

    Thread-safety invariant: every state-machine mutation and every database
    write therefore happens on the loop thread (the same thread the DB
    connection and the wearable loop use). The only cross-thread touch is the
    worker reading ``sm.state`` for ``health()`` — a GIL-atomic enum read,
    exactly as in the single-camera path. No worker thread touches ``repos``
    directly.

    Failure isolation: a persistence error, an alert error, or a loop stall
    (``timeout``) is logged and swallowed (returns ``[]``) so one camera's
    database or alert problem can never crash its worker or disturb the other
    camera. The finite ``timeout`` also guarantees the worker can never deadlock
    waiting on a wedged loop during shutdown.
    """

    def observe(evidence: bool, confidence: float) -> list[Transition]:
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(
                event_manager.observe(evidence, confidence), loop
            )
            return future.result(timeout=timeout)
        except Exception as exc:  # persist / alert / timeout — isolated per camera
            # On timeout the coroutine is still running on the loop; cancel it so a
            # wedged loop can never accumulate orphaned observe() tasks.
            if future is not None:
                future.cancel()
            log.warning(
                "camera %s event bridge isolated a %s (persist/alert not applied this frame)",
                camera_id, type(exc).__name__,
            )
            return []

    return observe


def build_multi_camera_monitor(
    settings: Any,
    camera_configs: list[CameraConfig],
    *,
    detector: FallDetector | None = None,
    clock: Clock | None = None,
    observe_factory: Callable[[str, FallEventStateMachine], ObserveFn] | None = None,
) -> MultiCameraMonitor:
    """Build a monitor with one shared detector and one RTSP worker per config.

    ``observe_factory(camera_id, state_machine)`` lets the app bridge each
    camera's observations to its async EventManager (persist + alert); when
    omitted, the worker drives its own state machine directly (the diagnostic).
    """
    from vytallink.vision.factory import build_detector
    from vytallink.vision.rtsp import RTSPCamera

    clock = clock or SystemClock()
    detector = detector or build_detector(settings, clock=clock)
    stale_timeout = max(2.0, 3.0 / max(0.1, settings.detect_max_fps))

    def factory(monitor: MultiCameraMonitor) -> list[CameraWorker]:
        workers: list[CameraWorker] = []
        for cfg in camera_configs:
            camera = RTSPCamera(cfg.rtsp_url(), source_id=cfg.camera_id, clock=clock, stale_timeout=stale_timeout)
            sm = FallEventStateMachine(
                confirm_seconds=settings.fall_confirm_seconds,
                clear_seconds=settings.fall_clear_seconds,
                cooldown_seconds=settings.alert_cooldown_seconds,
                source_device=cfg.camera_id,
                clock=clock,
                reconfirm_cooldown_seconds=settings.fall_reconfirm_cooldown_seconds,
            )
            observe_fn = observe_factory(cfg.camera_id, sm) if observe_factory else sm.observe
            workers.append(
                CameraWorker(
                    cfg,
                    camera,
                    sm,
                    observe_fn,
                    monitor.infer,
                    fall_class_set=settings.fall_class_set,
                    confidence_threshold=settings.confidence_threshold,
                    clock=clock,
                    evidence_smoother=FallEvidenceSmoother(settings.evidence_hold_seconds, clock=clock),
                    max_fps=settings.detect_max_fps,
                )
            )
        return workers

    return MultiCameraMonitor(detector, factory, clock=clock)
