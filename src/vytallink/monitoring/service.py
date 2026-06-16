"""MonitoringService — the runtime that wires providers to the event pipeline.

Two orchestration modes:

* **simulation** (``VISION_MODE=simulation``): the fall pipeline is driven
  deterministically and instantly by the simulation controls using a
  ``ManualClock``. A lightweight heartbeat loop still reads the simulated
  camera + detector so health/frame counters stay live, but it does NOT feed
  the state machine (so a simulated fall persists until cleared/reset). This
  exercises the *real* camera→detector→evidence→state-machine→alert pipeline;
  only the clock is advanced by hand instead of by wall time.

* **live** (``file``/``rtsp``): a real-time detection loop reads frames, runs
  the detector, and feeds evidence to the state machine on a ``SystemClock``.

A wearable loop runs in both modes, sampling the (simulated) wearable on a
timer and persisting readings. All provider exceptions are isolated so a single
failure never crashes the service.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from vytallink import __phase__, __version__
from vytallink.alerts.factory import build_dispatcher
from vytallink.common.clock import ManualClock, SystemClock, isoformat
from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus
from vytallink.config import Settings, VisionMode
from vytallink.database import Database, DeviceRow, Repositories, VitalRow
from vytallink.events import EventManager, FallEventStateMachine, FallState
from vytallink.monitoring import system_info
from vytallink.vision import build_camera, build_detector, detections_to_evidence
from vytallink.vision.detector_simulated import Scenario, SimulatedFallDetector
from vytallink.wearable import build_wearable

log = get_logger("monitoring.service")


class MonitoringService:
    def __init__(
        self,
        settings: Settings,
        *,
        db: Database | None = None,
        event_clock: Any | None = None,
    ) -> None:
        self.settings = settings
        self.system_clock = SystemClock()
        self.simulation_mode = settings.vision_mode == VisionMode.SIMULATION

        # In simulation, the event timeline is driven by a ManualClock so falls
        # can be confirmed instantly & deterministically. In live mode, the
        # real-time loop drives a SystemClock.
        if event_clock is not None:
            self.event_clock = event_clock
        elif self.simulation_mode:
            self.event_clock = ManualClock(start=self.system_clock.now())
        else:
            self.event_clock = self.system_clock

        self.db = db or Database(settings.database_path, clock=self.system_clock)
        self.repos = Repositories(self.db)
        self.dispatcher = build_dispatcher(settings, self.repos, clock=self.system_clock)
        self.state_machine = FallEventStateMachine(
            confirm_seconds=settings.fall_confirm_seconds,
            clear_seconds=settings.fall_clear_seconds,
            cooldown_seconds=settings.alert_cooldown_seconds,
            source_device=settings.camera_device_id,
            clock=self.event_clock,
        )
        self.event_manager = EventManager(
            self.repos,
            self.state_machine,
            self.dispatcher,
            clock=self.event_clock,
            simulated=self.simulation_mode,
        )
        self.camera = build_camera(settings, clock=self.system_clock)
        self.detector = build_detector(settings)
        self.wearable = build_wearable(settings, clock=self.system_clock)

        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._started_at: datetime | None = None
        self._last_inference_time: datetime | None = None
        self._last_vital: VitalRow | None = None
        self._sim_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self.settings.ensure_runtime_dirs()
        self.db.initialize()
        self.detector.load()
        self._register_devices()

        try:
            self.camera.open()
        except CameraError as exc:
            log.warning("Camera did not open at startup: %s", exc)

        try:
            self.wearable.connect()
            self._update_device(self.settings.wearable_device_id, HealthStatus.OK)
        except Exception as exc:
            log.warning("Wearable did not connect at startup: %s", exc)
            self._update_device(self.settings.wearable_device_id, HealthStatus.DOWN, error=str(exc))

        self._started_at = self.system_clock.now()
        self._running = True
        # Prime one wearable reading so the dashboard has immediate data.
        await self._sample_wearable_once()
        self._tasks = [
            asyncio.create_task(self._wearable_loop(), name="vytallink-wearable"),
            asyncio.create_task(self._monitor_loop(), name="vytallink-monitor"),
        ]
        log.info(
            "MonitoringService started (mode=%s, env=%s)",
            "simulation" if self.simulation_mode else self.settings.vision_mode.value,
            self.settings.env.value,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown best-effort
                pass
        self._tasks = []
        try:
            self.camera.close()
        except Exception:  # pragma: no cover - defensive
            pass
        self.wearable.disconnect()
        await self.dispatcher.aclose()
        self.db.close()
        log.info("MonitoringService stopped")

    # -- device registration ----------------------------------------------
    def _register_devices(self) -> None:
        self.repos.devices.upsert(
            DeviceRow(
                device_id=self.settings.camera_device_id,
                device_type="camera",
                display_name=self.camera.description,
                connection_status=self.camera.status().value,
                metadata={"mode": self.settings.vision_mode.value},
            )
        )
        self.repos.devices.upsert(
            DeviceRow(
                device_id=self.settings.wearable_device_id,
                device_type="wearable",
                display_name=self.wearable.display_name,
                connection_status=self.wearable.status().value,
                metadata={"mode": self.settings.wearable_mode.value, "simulated": True},
            )
        )

    def _update_device(
        self, device_id: str, status: HealthStatus, *, error: str | None = None, seen: bool = False
    ) -> None:
        fields: dict[str, Any] = {"connection_status": status.value}
        if seen:
            fields["last_seen"] = isoformat(self.system_clock.now())
        if error is not None:
            fields["last_error"] = error
        try:
            self.repos.devices.update(device_id, **fields)
        except Exception:  # pragma: no cover - device may not be registered yet
            pass

    # -- loops -------------------------------------------------------------
    async def _wearable_loop(self) -> None:
        interval = max(0.5, self.settings.wearable_sample_seconds)
        try:
            while self._running:
                await asyncio.sleep(interval)
                await self._sample_wearable_once()
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise

    async def _sample_wearable_once(self) -> None:
        try:
            reading = self.wearable.read()
        except Exception as exc:
            log.warning("Wearable read failed: %s", exc)
            self._update_device(self.settings.wearable_device_id, HealthStatus.DOWN, error=str(exc))
            return
        if reading is None:
            return
        row = self.repos.vitals.insert(
            VitalRow(
                timestamp=isoformat(reading.timestamp),
                device_id=reading.device_id,
                heart_rate=reading.heart_rate,
                motion=reading.motion,
                connection_quality=reading.connection_quality,
                battery=reading.battery,
                simulated=reading.simulated,
                metadata=reading.metadata,
            )
        )
        self._last_vital = row
        self._update_device(reading.device_id, HealthStatus.OK, seen=True)

    async def _monitor_loop(self) -> None:
        interval = max(0.05, self.settings.monitor_loop_interval)
        try:
            while self._running:
                await asyncio.sleep(interval)
                if self.simulation_mode:
                    self._heartbeat_once()  # health only; does not observe
                else:
                    await self._detect_and_observe_once()
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise

    def _read_frame_for_detection(self) -> Frame | None:
        frame = self.camera.read()
        if frame is None and self.simulation_mode:
            # Camera dropout in simulation: synthesize a frame so health/inference
            # timestamps keep flowing and scenario-based evidence still works.
            frame = Frame(
                frame_id=self.camera.frame_count + 1,
                timestamp=self.system_clock.now(),
                source_id=self.settings.camera_device_id,
            )
        return frame

    def _detect_once(self) -> tuple[bool, float]:
        frame = self._read_frame_for_detection()
        if frame is None:
            self._update_device(self.settings.camera_device_id, self.camera.status())
            return False, 0.0
        detections = self.detector.infer(frame)
        self._last_inference_time = self.system_clock.now()
        self._update_device(self.settings.camera_device_id, self.camera.status(), seen=True)
        return detections_to_evidence(
            detections, self.settings.fall_class_set, self.settings.confidence_threshold
        )

    def _heartbeat_once(self) -> None:
        # Read + infer for liveness/health, but do NOT feed the state machine.
        self._detect_once()

    async def _detect_and_observe_once(self) -> None:
        evidence, confidence = self._detect_once()
        await self.event_manager.observe(evidence, confidence)

    # -- simulation controls (deterministic, real pipeline) ----------------
    def _ensure_simulatable(self) -> None:
        if not self.simulation_mode or not isinstance(self.detector, SimulatedFallDetector):
            raise RuntimeError("Simulation controls require VISION_MODE=simulation")

    async def simulate_fall(self) -> dict[str, Any]:
        self._ensure_simulatable()
        async with self._sim_lock:
            self.event_clock.set_now(self.system_clock.now())  # anchor realistic timestamps
            self.detector.set_scenario(Scenario.FALL)
            evidence, conf = self._detect_once()
            await self.event_manager.observe(evidence, conf)  # NORMAL/RESOLVED -> POSSIBLE
            self.event_clock.advance(self.settings.fall_confirm_seconds + 0.05)
            evidence, conf = self._detect_once()
            await self.event_manager.observe(evidence, conf)  # -> CONFIRMED (+ one alert)
            return self.status()

    async def simulate_normal(self) -> dict[str, Any]:
        self._ensure_simulatable()
        async with self._sim_lock:
            self.detector.set_scenario(Scenario.NORMAL)
            evidence, conf = self._detect_once()
            await self.event_manager.observe(evidence, conf)  # CONFIRMED -> RECOVERING
            self.event_clock.advance(self.settings.fall_clear_seconds + 0.05)
            evidence, conf = self._detect_once()
            await self.event_manager.observe(evidence, conf)  # -> RESOLVED
            evidence, conf = self._detect_once()
            await self.event_manager.observe(evidence, conf)  # -> NORMAL
            return self.status()

    async def simulate_reset(self) -> dict[str, Any]:
        self._ensure_simulatable()
        async with self._sim_lock:
            self.detector.set_scenario(Scenario.NORMAL)
            await self.event_manager.reset()
            return self.status()

    # -- caregiver operations (pass-through) -------------------------------
    async def label_event(self, event_uid: str, label: str):
        return await self.event_manager.label_event(event_uid, label)

    async def resolve_event(self, event_uid: str, note: str | None = None):
        return await self.event_manager.resolve_event(event_uid, note)

    # -- status / health ---------------------------------------------------
    def uptime_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return round((self.system_clock.now() - self._started_at).total_seconds(), 1)

    def _detector_health(self) -> dict[str, Any]:
        try:
            return self.detector.health()
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": HealthStatus.UNKNOWN.value, "error": str(exc)}

    def _alert_health(self) -> dict[str, Any]:
        return {
            "status": HealthStatus.OK.value if self.dispatcher.providers else HealthStatus.DEGRADED.value,
            "providers": self.dispatcher.provider_names,
        }

    def health(self) -> dict[str, Any]:
        db_health = self.db.health()
        cam_health = self.camera.health()
        wear_health = self.wearable.health()
        disk = system_info.disk_info(self.settings.database_path, self.settings.disk_warning_percent)
        gpu = system_info.gpu_info()

        server_ok = self._running
        overall = HealthStatus.OK
        if not db_health.get("ok") or not server_ok:
            overall = HealthStatus.DOWN
        elif (
            cam_health["status"] == HealthStatus.DOWN.value
            or wear_health["status"] == HealthStatus.DOWN.value
            or disk.get("warning")
        ):
            overall = HealthStatus.DEGRADED

        return {
            "overall": overall.value,
            "version": __version__,
            "phase": __phase__,
            "server": {"status": HealthStatus.OK.value if server_ok else HealthStatus.DOWN.value, "running": server_ok},
            "database": {"status": HealthStatus.OK.value if db_health.get("ok") else HealthStatus.DOWN.value, **db_health},
            "camera": cam_health,
            "detector": self._detector_health(),
            "wearable": wear_health,
            "alerts": self._alert_health(),
            "gpu": gpu,
            "latest_frame_time": cam_health.get("last_frame_time"),
            "latest_inference_time": isoformat(self._last_inference_time),
            "fall_state": self.state_machine.state.value,
            "uptime_seconds": self.uptime_seconds(),
            "disk": disk,
            "disk_warning": bool(disk.get("warning")),
            "simulation": {
                "active": self.simulation_mode,
                "env": self.settings.env.value,
                "controls_enabled": self.controls_enabled(),
            },
        }

    def controls_enabled(self) -> bool:
        """Dev simulation controls are enabled only in development + simulation."""
        return self.settings.is_development and self.simulation_mode

    def status(self) -> dict[str, Any]:
        sm = self.event_manager.status()
        lv = self._last_vital
        return {
            "name": "VytalLink",
            "version": __version__,
            "phase": __phase__,
            "env": self.settings.env.value,
            "running": self._running,
            "uptime_seconds": self.uptime_seconds(),
            "fall_state": sm["fall_state"],
            "active_event_uid": sm["active_event_uid"],
            "current_confidence": sm["highest_confidence"],
            "counts": {
                "events": self.repos.events.count(),
                "alerts": self.repos.alerts.count(),
                "vitals": self.repos.vitals.count(),
            },
            "latest_vital": _vital_summary(lv),
            "camera_status": self.camera.status().value,
            "detector": self._detector_health().get("name"),
            "wearable_status": self.wearable.status().value,
            "gpu_available": system_info.gpu_info().get("available", False),
            "simulation_active": self.simulation_mode,
            "controls_enabled": self.controls_enabled(),
            "last_update": isoformat(self.system_clock.now()),
        }


def _vital_summary(v: VitalRow | None) -> dict[str, Any] | None:
    if v is None:
        return None
    return {
        "timestamp": v.timestamp,
        "device_id": v.device_id,
        "heart_rate": v.heart_rate,
        "motion": v.motion,
        "battery": v.battery,
        "connection_quality": v.connection_quality,
        "simulated": v.simulated,
    }
