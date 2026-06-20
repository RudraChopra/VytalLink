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
import hmac
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from vytallink import __phase__, __version__
from vytallink.alerts.factory import build_dispatcher
from vytallink.common.clock import ManualClock, SystemClock, isoformat
from vytallink.common.device import device_label
from vytallink.common.errors import CameraError
from vytallink.common.logging_setup import get_logger
from vytallink.common.types import Frame, HealthStatus, RawDetection
from vytallink.config import Settings, VisionMode
from vytallink.database import Database, DeviceRow, Repositories, VitalRow
from vytallink.database.models import IncidentVitalRow
from vytallink.events import EventManager, FallEventStateMachine, FallState
from vytallink.monitoring import system_info
from vytallink.vision import build_camera, build_detector, detections_to_evidence
from vytallink.vision.detector_simulated import Scenario, SimulatedFallDetector
from vytallink.vision.multi_camera import build_multi_camera_monitor, make_event_bridge
from vytallink.monitoring.alert_score import ScoreThresholds
from vytallink.monitoring.freshness import FreshnessThresholds, camera_freshness, camera_is_fresh
from vytallink.monitoring.incident_reconcile import classify_incident
from vytallink.monitoring.patient_state import build_patient_state
from vytallink.vision.evidence import FallEvidenceSmoother
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
        # Multi-camera mode is active when any CAMERA_{N}_* camera is enabled. It
        # overrides simulation: real RTSP cameras run through MultiCameraMonitor.
        self._camera_configs = settings.configured_cameras()
        self.multi_camera_mode = bool(self._camera_configs)
        self.simulation_mode = (
            settings.vision_mode == VisionMode.SIMULATION and not self.multi_camera_mode
        )
        self._multi_monitor = None  # built in start() when multi_camera_mode
        # One EventManager per camera (built in start(), on the loop). Each wraps
        # that camera's own state machine (source_device=camera_id) and SHARES the
        # one repos + dispatcher, so multi-camera events persist + alert through
        # the exact same path as the single-camera mode — no parallel system.
        self._camera_event_managers: dict[str, EventManager] = {}

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
        # Synthetic fall testing (validation-only) forces alerts to dry-run so a
        # forced fall can never page a real caregiver, and tags persisted events.
        self.synthetic_mode = settings.synthetic_detection_active
        self.dispatcher = build_dispatcher(
            settings, self.repos, clock=self.system_clock, dry_run=self.synthetic_mode
        )
        self.state_machine = FallEventStateMachine(
            confirm_seconds=settings.fall_confirm_seconds,
            clear_seconds=settings.fall_clear_seconds,
            cooldown_seconds=settings.alert_cooldown_seconds,
            source_device=settings.camera_device_id,
            clock=self.event_clock,
            reconfirm_cooldown_seconds=settings.fall_reconfirm_cooldown_seconds,
        )
        # Incident vitals snapshot writer: counters + a single failure-isolated
        # callback reused by every EventManager (one snapshot per incident).
        self._snapshot_count = 0
        self._snapshot_failures = 0
        self._snapshot_fn = self._make_snapshot_fn()
        # Stale-incident reconciliation counters (surfaced in /health, no details).
        self._incidents_reconciled = 0
        self._reconcile_failures = 0
        self._reconcile_ambiguous_open = 0
        self.event_manager = EventManager(
            self.repos,
            self.state_machine,
            self.dispatcher,
            clock=self.event_clock,
            simulated=self.simulation_mode,
            synthetic=self.synthetic_mode,
            snapshot_fn=self._snapshot_fn,
        )
        self.camera = build_camera(settings, clock=self.system_clock)
        self.detector = build_detector(settings, clock=self.system_clock)
        self.wearable = build_wearable(settings, clock=self.system_clock)
        # Live-only: bridge brief detection gaps. Simulation is deterministic and
        # never drops frames, so it needs (and gets) no smoothing.
        self._evidence_smoother = (
            None
            if self.simulation_mode
            else FallEvidenceSmoother(settings.evidence_hold_seconds, clock=self.system_clock)
        )

        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._started_at: datetime | None = None
        self._last_inference_time: datetime | None = None
        self._last_vital: VitalRow | None = None
        self._sim_lock = asyncio.Lock()
        # Single dedicated thread for ALL accelerator work (model load, warmup,
        # every inference). MPS/Metal command buffers are not safe across threads;
        # asyncio.to_thread's multi-worker pool intermittently aborts the process
        # with a Metal "scheduled handler after commit" assertion. Created in
        # start(); inference is pinned here so it never crosses threads.
        self._infer_executor: ThreadPoolExecutor | None = None
        # Latest decoded camera frame (BGR ndarray) the DETECTOR last processed —
        # used for the annotated preview (boxes align with this frame). The raw
        # live feed serves the camera's freshest frame directly (see
        # latest_frame_jpeg), decoupled from the detection loop. In-memory only —
        # never written to disk.
        self._last_frame_image: Any | None = None

        # --- live-detection pacing / debug state ---------------------------
        self._last_processed_seq: int | None = None  # de-dup: skip re-sent frames
        self._frames_dropped_stale = 0   # intentionally dropped (too old) — NOT failed reads
        self._frames_processed = 0       # frames that actually ran inference
        self._frames_with_fallen = 0
        self._class_counts: dict[str, int] = {}
        self._last_detections: list[RawDetection] = []
        self._last_detection_summary: list[dict[str, Any]] = []
        self._last_evidence: bool = False
        self._last_evidence_score: float = 0.0
        self._last_frame_age: float = 0.0
        self._transitions: deque[dict[str, Any]] = deque(maxlen=25)

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self.settings.ensure_runtime_dirs()
        self.db.initialize()

        if self.synthetic_mode:
            log.warning(
                "================ SYNTHETIC FALL TESTING ACTIVE ================ "
                "Non-fall postures are being treated as falls for validation; "
                "external alerts are DRY-RUN (no caregiver delivery) and every "
                "event is tagged event_type='fall_synthetic'. NOT FOR PRODUCTION."
            )

        if self.multi_camera_mode:
            # Multi-camera: one shared model + one inference lane (owned by the
            # monitor), N isolated RTSP workers. Each worker's observations are
            # bridged to a per-camera EventManager so confirmed falls persist to
            # the DB and dispatch alerts through the SAME path single-camera uses.
            loop = asyncio.get_running_loop()
            self._camera_event_managers = {}

            def _observe_factory(camera_id: str, sm: FallEventStateMachine):
                # Reuse the existing EventManager: same repos (one locked
                # connection) and same dispatcher, with this camera's own state
                # machine (source_device=camera_id, so every persisted event and
                # AlertEvent is tagged with the camera). observe() runs on the
                # loop; the bridge isolates any persist/alert failure per camera.
                em = EventManager(
                    self.repos, sm, self.dispatcher,
                    clock=self.system_clock, simulated=False, synthetic=self.synthetic_mode,
                    snapshot_fn=self._snapshot_fn,
                )
                self._camera_event_managers[camera_id] = em
                return make_event_bridge(em, loop, camera_id=camera_id)

            # Built here so the running loop is captured for the bridge; started
            # off the loop so the model load (seconds) never blocks startup.
            self._multi_monitor = build_multi_camera_monitor(
                self.settings, self._camera_configs, detector=self.detector,
                clock=self.system_clock, observe_factory=_observe_factory,
            )
            await asyncio.to_thread(self._multi_monitor.start)
            self._register_devices()
            self._started_at = self.system_clock.now()
            self._running = True
            self._connect_wearable_safe()
            await self._sample_wearable_once()
            await self._maybe_reconcile_on_startup()
            self._tasks = [
                asyncio.create_task(self._wearable_loop(), name="vytallink-wearable"),
                asyncio.create_task(self._reconcile_loop(), name="vytallink-reconcile"),
            ]
            log.info(
                "MonitoringService started (mode=rtsp_multi, cameras=%d, env=%s)",
                len(self._camera_configs), self.settings.env.value,
            )
            return

        # Pin model load + warmup to the dedicated inference thread so every later
        # inference runs on the SAME thread (MPS/Metal thread-affinity).
        self._infer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vytallink-infer"
        )
        await asyncio.get_running_loop().run_in_executor(
            self._infer_executor, self.detector.load
        )
        self._register_devices()

        try:
            self.camera.open()
        except CameraError as exc:
            log.warning("Camera did not open at startup: %s", exc)

        self._connect_wearable_safe()

        self._started_at = self.system_clock.now()
        self._running = True
        # Prime one wearable reading so the dashboard has immediate data.
        await self._sample_wearable_once()
        await self._maybe_reconcile_on_startup()
        self._tasks = [
            asyncio.create_task(self._wearable_loop(), name="vytallink-wearable"),
            asyncio.create_task(self._monitor_loop(), name="vytallink-monitor"),
            asyncio.create_task(self._reconcile_loop(), name="vytallink-reconcile"),
        ]
        log.info(
            "MonitoringService started (mode=%s, env=%s)",
            "simulation" if self.simulation_mode else self.settings.vision_mode.value,
            self.settings.env.value,
        )

    def _connect_wearable_safe(self) -> None:
        try:
            self.wearable.connect()
            self._update_device(self.settings.wearable_device_id, HealthStatus.OK)
        except Exception as exc:
            log.warning("Wearable did not connect at startup: %s", exc)
            self._update_device(self.settings.wearable_device_id, HealthStatus.DOWN, error=str(exc))

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
        if self._multi_monitor is not None:
            try:
                await asyncio.to_thread(self._multi_monitor.stop)
            except Exception:  # pragma: no cover - shutdown best-effort
                pass
        else:
            try:
                self.camera.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self.wearable.disconnect()
        await self.dispatcher.aclose()
        self.db.close()
        if self._infer_executor is not None:
            self._infer_executor.shutdown(wait=False)
            self._infer_executor = None
        log.info("MonitoringService stopped")

    # -- device registration ----------------------------------------------
    def _register_devices(self) -> None:
        if self.multi_camera_mode:
            for cfg in self._camera_configs:
                self.repos.devices.upsert(
                    DeviceRow(
                        device_id=cfg.camera_id,
                        device_type="camera",
                        display_name=cfg.safe_label(),     # host:port, no creds
                        connection_status=HealthStatus.UNKNOWN.value,
                        metadata={"mode": "rtsp_multi"},
                    )
                )
        else:
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

    # -- iPhone vitals ingestion + normalized patient state ----------------
    def ingest_vitals(self, payload: Any) -> tuple[VitalRow, bool]:
        """Validate timing + store one iPhone vitals sample (a validated
        VitalsIngest). Returns (row, idempotent). Raises ValueError on an
        unparseable / future / too-old timestamp. Never logs values/full payload.

        NOTE: there was no prior iPhone contract in this repo — POST /api/vitals
        and this payload shape are defined here and unverified against a device.
        """
        now = self.system_clock.now()
        source_ts = self._parse_source_timestamp(getattr(payload, "timestamp", None), now)
        device_id = (getattr(payload, "device_id", None) or "iphone-1")[:64]
        sample_id = getattr(payload, "sample_id", None)
        rr = getattr(payload, "respiratory_rate", None)
        posture = getattr(payload, "posture", None)
        # Idempotency: a retried sample (same device + sample_id) is not re-stored.
        if sample_id:
            for existing in self.repos.vitals.list(limit=25, device_id=device_id):
                if (existing.metadata or {}).get("sample_id") == sample_id:
                    return existing, True
        metadata = {
            "source": "iphone",
            "received_at": isoformat(now),
            "respiratory_rate": rr,
            "posture": posture,
            "phone_alert_score": getattr(payload, "phone_alert_score", None),
            "activity": getattr(payload, "motion", None),
            "sample_id": sample_id,
            # Which contract form the sender used (canonical vs legacy aliases) —
            # safe metadata, no values; helps reconcile the real device later.
            "contract_form": getattr(payload, "contract_form", "canonical"),
        }
        row = self.repos.vitals.insert(
            VitalRow(
                timestamp=isoformat(source_ts), device_id=device_id,
                heart_rate=getattr(payload, "heart_rate", None),
                motion=getattr(payload, "motion", None),
                connection_quality=None, battery=getattr(payload, "battery", None),
                simulated=False, metadata=metadata,
            )
        )
        self._last_vital = row
        signals = "+".join(
            k for k, val in (("hr", row.heart_rate), ("rr", rr), ("motion", row.motion), ("posture", posture))
            if val is not None
        )
        log.info(
            "Ingested iPhone vitals device=%s signals=[%s] age=%.1fs",  # safe: no values
            device_id, signals, max(0.0, (now - source_ts).total_seconds()),
        )
        return row, False

    def _parse_source_timestamp(self, ts_str: str | None, now: datetime) -> datetime:
        if not ts_str:
            return now
        try:
            parsed = datetime.fromisoformat(str(ts_str).strip().replace("Z", "+00:00"))
        except Exception:
            raise ValueError("timestamp is not valid ISO-8601")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = (parsed - now).total_seconds()
        if delta > self.settings.vitals_max_future_skew_seconds:
            raise ValueError("timestamp is in the future")
        if -delta > self.settings.vitals_reject_older_than_seconds:
            raise ValueError("timestamp is too old")
        return parsed

    @staticmethod
    def _parse_iso(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _active_camera_ids(self) -> set[str]:
        """Camera ids configured for the running mode (multi vs single/sim). An
        incident whose source_device is NOT in this set is orphaned."""
        if self.multi_camera_mode:
            return {cfg.camera_id for cfg in self._camera_configs}
        return {self.settings.camera_device_id}

    async def _reconcile_incidents(self, trigger: str) -> dict[str, int]:
        """Auto-resolve stale, unsupported incidents via the supported resolve
        path (no new alert; snapshot untouched; history preserved). Idempotent.
        A configured-but-offline source camera leaves the incident OPEN.

        ``trigger`` is 'startup_reconciliation' or 'runtime'. Never raises.
        """
        result = {"resolved": 0, "ambiguous": 0, "failures": 0}
        if not self.settings.incident_auto_resolve_enabled:
            self._reconcile_ambiguous_open = 0
            return result
        try:
            rows = self.db.query_all(
                "SELECT event_uid, source_device, state, updated_at, confirmed_time "
                "FROM events WHERE state IN ('confirmed_fall','recovering')"
            )
        except Exception as exc:  # pragma: no cover - defensive; never crash monitoring
            self._reconcile_failures += 1
            log.warning("incident reconciliation query failed: %s", type(exc).__name__)
            return result

        active_ids = self._active_camera_ids()
        summaries = {s["id"]: s for s in self._camera_summaries()}
        fresh_thr = FreshnessThresholds.from_settings(self.settings)
        now = self.system_clock.now()
        for r in rows:
            src = r["source_device"]
            last = self._parse_iso(r["updated_at"]) or self._parse_iso(r["confirmed_time"])
            age = (now - last).total_seconds() if last else float("inf")
            summary = summaries.get(src)
            cam_fresh, cam_state = False, None
            if summary is not None:
                cam_state = summary.get("fall_state")
                cam_fresh = camera_is_fresh(camera_freshness(
                    summary.get("frame_age_seconds"), connected=bool(summary.get("connected")), t=fresh_thr))
            resolve, reason, ambiguous = classify_incident(
                age_seconds=age, stale_seconds=self.settings.incident_stale_seconds,
                source_in_config=(src in active_ids), camera_fresh=cam_fresh, camera_fall_state=cam_state,
            )
            if ambiguous:
                result["ambiguous"] += 1
            if not resolve:
                continue
            try:
                await self.event_manager.resolve_event(
                    r["event_uid"], note=f"auto-reconciled[{trigger}]: {reason}"
                )
                self._incidents_reconciled += 1
                result["resolved"] += 1
                log.info("incident reconciled event=%s trigger=%s reason=%s", r["event_uid"], trigger, reason)
            except Exception as exc:  # isolation: a resolve failure never crashes monitoring
                self._reconcile_failures += 1
                result["failures"] += 1
                log.warning("incident reconcile failed for %s: %s", r["event_uid"], type(exc).__name__)
        self._reconcile_ambiguous_open = result["ambiguous"]
        return result

    async def _maybe_reconcile_on_startup(self) -> None:
        if self.settings.incident_reconcile_on_startup:
            res = await self._reconcile_incidents("startup_reconciliation")
            log.info("startup incident reconciliation: %s", res)

    async def _reconcile_loop(self) -> None:
        interval = max(5.0, self.settings.incident_reconcile_interval_seconds)
        try:
            while self._running:
                await asyncio.sleep(interval)
                await self._reconcile_incidents("runtime")
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise

    def _active_incident_id(self) -> str | None:
        """Most recent unresolved incident (confirmed/recovering), or None."""
        try:
            row = self.db.query_one(
                "SELECT event_uid FROM events WHERE state IN ('confirmed_fall','recovering') "
                "ORDER BY created_at DESC LIMIT 1"
            )
        except Exception:  # pragma: no cover - defensive
            return None
        return row["event_uid"] if row else None

    def _camera_summaries(self) -> list[dict[str, Any]]:
        """Per-camera primitives for patient-state aggregation (mode-agnostic)."""
        out: list[dict[str, Any]] = []
        if self.multi_camera_mode and self._multi_monitor is not None:
            for cid, c in self._multi_monitor.health().get("cameras", {}).items():
                age_ms = c.get("last_frame_age_ms")
                out.append({
                    "id": cid, "connected": bool(c.get("connected")),
                    "fall_state": c.get("fall_state", "normal"),
                    "frame_age_seconds": (age_ms / 1000.0) if age_ms is not None else None,
                    "person_count": c.get("person_count"), "fall_confidence": None,
                })
        else:
            ch = self.camera.health()
            age = ch.get("last_frame_age_seconds")
            if self.simulation_mode and age is None:
                age = 0.0  # simulation synthesizes frames continuously -> fresh
            out.append({
                "id": self.settings.camera_device_id,
                "connected": bool(ch.get("opened", self.simulation_mode)),
                "fall_state": self.state_machine.state.value,
                "frame_age_seconds": age, "person_count": None,
                "fall_confidence": self._last_evidence_score or None,
            })
        return out

    # -- incident vitals snapshot (one per confirmed incident) -------------
    def _make_snapshot_fn(self):
        """A failure-isolated callback that captures one vitals snapshot when an
        incident is first confirmed. Runs on the event loop (same thread as all
        DB writes); never raises into observe/persistence/the camera worker."""

        def snapshot(event: Any) -> None:
            try:
                self._write_incident_snapshot(event)
            except Exception as exc:  # isolation: snapshot errors never propagate
                self._snapshot_failures += 1
                log.warning(
                    "incident snapshot failed for %s: %s",
                    getattr(event, "event_uid", "?"), type(exc).__name__,
                )

        return snapshot

    def _write_incident_snapshot(self, event: Any) -> None:
        ps = self.patient_state()
        vit, alert = ps["vitals"], ps["alert"]
        v = self.repos.vitals.latest()
        sample_id = (v.metadata or {}).get("sample_id") if v is not None else None
        confirmed = getattr(event, "confirmed_time", None)
        snap = IncidentVitalRow(
            event_uid=event.event_uid,
            camera_id=getattr(event, "source_device", None) or self.settings.camera_device_id,
            confirmed_time=isoformat(confirmed) if confirmed else None,
            vitals_sample_id=sample_id,
            heart_rate=vit.get("heart_rate"),
            respiratory_rate=vit.get("respiratory_rate"),
            posture=vit.get("posture"),
            phone_alert_score=vit.get("phone_alert_score"),
            computed_alert_level=alert.get("level"),
            computed_alert_score=alert.get("score"),
            reason_codes=list(alert.get("reasons", [])),
            source_timestamp=vit.get("source_timestamp"),
            received_at=vit.get("received_at"),
            vitals_age_seconds=vit.get("age_seconds"),
            vitals_freshness=vit.get("freshness"),
            # Available = a vital exists and is not so old it's "unavailable".
            vitals_available=(v is not None and vit.get("freshness") != "unavailable"),
            vitals_source=vit.get("source"),
            synthetic=self.synthetic_mode,
        )
        _row, created = self.repos.incident_vitals.create(snap)
        if created:
            self._snapshot_count += 1
            log.info(
                "incident snapshot stored event=%s camera=%s vitals_freshness=%s available=%s synthetic=%s",
                snap.event_uid, snap.camera_id, snap.vitals_freshness, snap.vitals_available, snap.synthetic,
            )

    def patient_state(self) -> dict[str, Any]:
        """Normalized patient state: latest vitals + per-camera fall state +
        freshness + informational alert score (raw vs computed distinguishable)."""
        now = self.system_clock.now()
        v = self.repos.vitals.latest()
        source_ts = received_at = None
        if v is not None:
            source_ts = self._parse_iso(v.timestamp)
            md = v.metadata or {}
            received_at = self._parse_iso(md.get("received_at") or v.created_at)
        return build_patient_state(
            now=now, vital=v, received_at=received_at, source_timestamp=source_ts,
            cameras=self._camera_summaries(), active_incident_id=self._active_incident_id(),
            fresh_thr=FreshnessThresholds.from_settings(self.settings),
            score_thr=ScoreThresholds.from_settings(self.settings),
        )

    async def _monitor_loop(self) -> None:
        if self.simulation_mode:
            await self._simulation_heartbeat_loop()
        else:
            await self._live_detection_loop()

    async def _simulation_heartbeat_loop(self) -> None:
        interval = max(0.05, self.settings.monitor_loop_interval)
        try:
            while self._running:
                await asyncio.sleep(interval)
                self._heartbeat_once()  # health only; does not observe
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise

    async def _live_detection_loop(self) -> None:
        """Live mode: run inference on the FRESHEST frame, paced to detect_max_fps.

        No fixed per-frame sleep — the loop runs as fast as inference allows up to
        the cap, so a fast model is not throttled. Camera ingest runs in the
        provider's own grabber thread, so detection never blocks ingest; the
        dashboard encodes off this loop (in the request handler), so it never
        blocks inference.
        """
        min_interval = 1.0 / max(0.1, self.settings.detect_max_fps)
        try:
            while self._running:
                t0 = self.system_clock.monotonic()
                await self._detect_and_observe_once()
                dt = self.system_clock.monotonic() - t0
                # Pace to the cap; if there is no new frame the loop idles here.
                await asyncio.sleep(max(0.0, min_interval - dt))
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
        """Simulation path: read + infer the (scenario-driven) simulated frame."""
        frame = self._read_frame_for_detection()
        if frame is None:
            self._update_device(self.settings.camera_device_id, self.camera.status())
            return False, 0.0
        if frame.image is not None:
            self._last_frame_image = frame.image
        detections = self.detector.infer(frame)
        if getattr(self.detector, "last_infer_ok", True):
            self._last_inference_time = self.system_clock.now()
        self._update_device(self.settings.camera_device_id, self.camera.status(), seen=True)
        evidence, confidence = detections_to_evidence(
            detections, self.settings.fall_class_set, self.settings.confidence_threshold
        )
        if self._evidence_smoother is not None:
            fall_set = self.settings.fall_class_set
            had_detection = bool(detections)
            had_upright = any(d.class_name.lower() not in fall_set for d in detections)
            evidence, confidence = self._evidence_smoother.update(
                evidence, confidence, had_detection=had_detection, had_upright=had_upright
            )
        return evidence, confidence

    def _detect_once_live(self) -> tuple[bool, float] | None:
        """Live path on the FRESHEST frame. Returns the (evidence, confidence) to
        observe, or ``None`` when there is no new fresh frame to act on this tick
        (no frame / duplicate / intentionally dropped because too old)."""
        cam_id = self.settings.camera_device_id
        if not getattr(self.camera, "has_latest_buffer", False):
            # Sequential source (e.g. a video file) — read frames in order.
            frame = self.camera.read()
            self._update_device(cam_id, self.camera.status(), seen=frame is not None)
            if frame is None or frame.image is None:
                return None
            self._last_frame_age = 0.0
            return self._infer_and_score(frame, frame.image)

        # Drive the provider's liveness + bounded-backoff reconnection + counters
        # without letting it pace detection: read() returns the latest (or None on
        # a real stall) and reopens a dead grabber. We do NOT reconnect just
        # because we poll slowly — the loop polls fast and the grabber owns ingest.
        self.camera.read()
        peek = self.camera.peek_latest()
        if peek is None:
            # Buffered source with no frame yet (e.g. opening / reconnecting).
            self._update_device(cam_id, self.camera.status(), seen=False)
            return None
        self._update_device(cam_id, self.camera.status(), seen=True)
        image, seq, age = peek
        self._last_frame_age = round(age, 3)
        # De-dup: the relay may re-send the same frame; only process new captures.
        if seq == self._last_processed_seq:
            return None
        self._last_processed_seq = seq
        # Drop a stale frame BEFORE inference — always work on fresh frames.
        if age > self.settings.detect_max_frame_age_seconds:
            self._frames_dropped_stale += 1
            return None

        h, w = (image.shape[0], image.shape[1]) if hasattr(image, "shape") else (0, 0)
        frame = Frame(
            frame_id=seq,
            timestamp=self.system_clock.now(),
            source_id=cam_id,
            width=int(w),
            height=int(h),
            image=image,
        )
        return self._infer_and_score(frame, image)

    def _infer_and_score(self, frame: Frame, image: Any) -> tuple[bool, float]:
        detections = self.detector.infer(frame)
        if getattr(self.detector, "last_infer_ok", True):
            self._last_inference_time = self.system_clock.now()
        # Keep the processed frame + its detections for the annotated preview
        # (boxes align with THIS frame). Never run the detector twice for drawing.
        self._last_frame_image = image
        self._last_detections = detections
        self._record_detection_debug(detections)

        evidence, confidence = detections_to_evidence(
            detections, self.settings.fall_class_set, self.settings.confidence_threshold
        )
        if self._evidence_smoother is not None:
            fall_set = self.settings.fall_class_set
            had_detection = bool(detections)
            had_upright = any(d.class_name.lower() not in fall_set for d in detections)
            evidence, confidence = self._evidence_smoother.update(
                evidence, confidence, had_detection=had_detection, had_upright=had_upright
            )
        self._last_evidence = evidence
        self._last_evidence_score = round(float(confidence), 4)
        return evidence, confidence

    def _record_detection_debug(self, detections: list[RawDetection]) -> None:
        self._frames_processed += 1
        fall_set = self.settings.fall_class_set
        summary: list[dict[str, Any]] = []
        saw_fallen = False
        for d in detections:
            name = d.class_name.lower()
            self._class_counts[name] = self._class_counts.get(name, 0) + 1
            md = d.metadata or {}
            summary.append({
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                # Normalized geometry for false-positive analysis (no pixels).
                "bbox_norm": md.get("bbox_norm"),
                "area_frac": md.get("area_frac"),
                "aspect": md.get("aspect"),
                "vertical_center": md.get("vertical_center"),
                "edges": md.get("edges"),
                "rejection": md.get("rejection"),
                "raw_class": md.get("raw_class"),
            })
            if name in fall_set:
                saw_fallen = True
        if saw_fallen:
            self._frames_with_fallen += 1
        # Newest first; cap the per-frame summary so it stays small/safe.
        self._last_detection_summary = summary[:8]

    def _heartbeat_once(self) -> None:
        # Read + infer for liveness/health, but do NOT feed the state machine.
        self._detect_once()

    async def _detect_and_observe_once(self) -> None:
        # Camera read + model inference are blocking native calls (cv2 / torch).
        # Run on the SINGLE dedicated inference thread (not asyncio.to_thread's
        # multi-worker pool) so MPS/Metal work never crosses threads. A slow
        # source still cannot freeze the event loop; observe() + transition
        # recording stay on the loop.
        executor = self._infer_executor
        result = await asyncio.get_running_loop().run_in_executor(
            executor, self._detect_once_live
        )
        if result is None:
            return  # no new fresh frame this tick
        evidence, confidence = result
        transitions = await self.event_manager.observe(evidence, confidence)
        for t in transitions:
            self._transitions.append(
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "reason": getattr(t.reason, "value", str(t.reason)),
                    "time": isoformat(t.timestamp),
                }
            )

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
        if not self.settings.alerts_enabled:
            # Intentionally off (e.g. during a live hardware test) — report
            # DISABLED, not DEGRADED, so it is clearly deliberate.
            return {"status": HealthStatus.DISABLED.value, "providers": []}
        return {
            "status": HealthStatus.OK.value if self.dispatcher.providers else HealthStatus.DEGRADED.value,
            "providers": self.dispatcher.provider_names,
        }

    def _aggregate_camera_health(self, vision: dict[str, Any]) -> dict[str, Any]:
        """Summarize per-camera health into the legacy ``camera`` block + a
        worst-of status used by the overall-health computation."""
        cams = vision.get("cameras", {})
        statuses = [c.get("status") for c in cams.values()]
        connected = [bool(c.get("connected")) for c in cams.values()]
        if not cams:
            agg = HealthStatus.DOWN.value
        elif not any(connected):
            agg = HealthStatus.DOWN.value
        elif not all(connected) or any(s in (None, "down") for s in statuses) or any(s == "degraded" for s in statuses):
            agg = HealthStatus.DEGRADED.value
        else:
            agg = HealthStatus.OK.value
        return {
            "status": agg,
            "description": f"{len(cams)} RTSP camera(s)",
            "cameras_connected": sum(connected),
            "cameras_total": len(cams),
        }

    def health(self) -> dict[str, Any]:
        db_health = self.db.health()
        vision: dict[str, Any] | None = None
        if self.multi_camera_mode and self._multi_monitor is not None:
            vision = self._multi_monitor.health()
            cam_health = self._aggregate_camera_health(vision)
        else:
            cam_health = self.camera.health()
            # Intentionally-dropped (too-old) frames are tracked by the service, not
            # the provider; surface them alongside the provider's failed-read count so
            # the two are never conflated on the dashboard.
            if not self.simulation_mode:
                cam_health = {
                    **cam_health,
                    "frames_processed": self._frames_processed,
                    "frames_dropped_stale": self._frames_dropped_stale,
                }
        wear_health = self.wearable.health()
        det_health = self._detector_health()
        disk = system_info.disk_info(self.settings.database_path, self.settings.disk_warning_percent)
        gpu = system_info.gpu_info()

        server_ok = self._running
        live = not self.simulation_mode
        det_status = det_health.get("status")
        overall = HealthStatus.OK
        if not db_health.get("ok") or not server_ok:
            overall = HealthStatus.DOWN
        elif live and det_status == HealthStatus.DOWN.value:
            # In live mode a non-functional detector means no fall can be detected.
            overall = HealthStatus.DOWN
        elif (
            cam_health["status"] == HealthStatus.DOWN.value
            or wear_health["status"] == HealthStatus.DOWN.value
            or disk.get("warning")
            or det_status == HealthStatus.DEGRADED.value
            or (live and cam_health["status"] == HealthStatus.DEGRADED.value)
            # An ambiguous open incident (source camera offline) needs attention.
            or self._reconcile_ambiguous_open > 0
            or self._reconcile_failures > 0
        ):
            overall = HealthStatus.DEGRADED

        if self.multi_camera_mode:
            mode = "rtsp_multi"
        elif self.simulation_mode:
            mode = "simulation"
        else:
            mode = self.settings.vision_mode.value
        # In multi-camera mode there is no single state machine; report the worst
        # (most-advanced) fall state across cameras.
        if vision is not None:
            states = [c.get("fall_state", "normal") for c in vision.get("cameras", {}).values()]
            fall_state = _worst_fall_state(states)
        else:
            fall_state = self.state_machine.state.value
        payload = {
            "overall": overall.value,
            "version": __version__,
            "phase": __phase__,
            "mode": mode,
            "camera_name": cam_health.get("safe_source") or cam_health.get("description"),
            "server": {"status": HealthStatus.OK.value if server_ok else HealthStatus.DOWN.value, "running": server_ok},
            "database": {"status": HealthStatus.OK.value if db_health.get("ok") else HealthStatus.DOWN.value, **db_health},
            "camera": cam_health,
            "detector": det_health,
            "wearable": wear_health,
            "alerts": self._alert_health(),
            "gpu": gpu,
            "latest_frame_time": cam_health.get("last_frame_time"),
            "latest_inference_time": isoformat(self._last_inference_time),
            "fall_state": fall_state,
            "uptime_seconds": self.uptime_seconds(),
            "disk": disk,
            "disk_warning": bool(disk.get("warning")),
            "simulation": {
                "active": self.simulation_mode,
                "env": self.settings.env.value,
                "controls_enabled": self.controls_enabled(),
            },
            "live_video": self.live_video_enabled(),
            # Whether a token is required to view the feed (NOT the token itself).
            "video_protected": self.video_token_required(),
        }
        payload["model"] = self._model_block(det_health)
        payload["startup"] = self._startup_block()
        payload["synthetic_detection_mode"] = self.synthetic_mode
        payload["persistence"] = self._persistence_block()
        if vision is not None:
            payload["vision"] = vision   # mode=rtsp_multi + per-camera (credential-free)
        return payload

    def _model_load_count(self) -> int:
        if self._multi_monitor is not None:
            return getattr(self._multi_monitor, "model_load_count", 1)
        return 1 if self._detector_health().get("loaded") else 0

    def _model_block(self, det: dict[str, Any]) -> dict[str, Any]:
        """Explicit model-lifecycle view for /health (credential-free).

        states: ready | degraded | failed | loading. The model is never 'ready'
        until it is loaded; a load error surfaces as 'failed' and (in live mode)
        degrades overall health so consumers never treat it as usable."""
        loaded = bool(det.get("loaded"))
        status = det.get("status")
        last_error = det.get("last_error")
        if loaded and status == HealthStatus.OK.value:
            state = "ready"
        elif loaded and status == HealthStatus.DEGRADED.value:
            state = "degraded"
        elif last_error and not loaded:
            state = "failed"
        elif not loaded:
            state = "loading"
        else:
            state = status
        return {
            "state": state,
            "device": det.get("device"),
            "load_count": self._model_load_count(),
            "warmup_complete": det.get("warmup_ms") is not None,
            "last_error": last_error,
        }

    def _persistence_block(self) -> dict[str, Any]:
        """Event/snapshot writer health (no patient values — counts only)."""
        try:
            total = self.repos.incident_vitals.count()
        except Exception:  # pragma: no cover - defensive
            total = None
        return {
            "snapshot_writer": HealthStatus.DEGRADED.value if self._snapshot_failures else HealthStatus.OK.value,
            "snapshots_written": self._snapshot_count,
            "snapshot_failures": self._snapshot_failures,
            "incident_snapshots_total": total,
            "reconciliation": {
                "auto_resolve_enabled": self.settings.incident_auto_resolve_enabled,
                "incidents_reconciled": self._incidents_reconciled,
                "reconcile_failures": self._reconcile_failures,
                "open_ambiguous": self._reconcile_ambiguous_open,
            },
        }

    def _startup_block(self) -> dict[str, Any]:
        """Safe startup/retry status. attempt/max are set by the start wrapper
        (start.sh) via env; absent -> a direct run, attempt 1."""
        import os

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, "") or default)
            except ValueError:
                return default

        return {
            "attempt": _int("STARTUP_ATTEMPT", 1),
            "max_attempts": _int("STARTUP_MAX_ATTEMPTS", 1),
            "completed": bool(self._running),
            "uptime_seconds": self.uptime_seconds(),
        }

    def controls_enabled(self) -> bool:
        """Dev simulation controls are enabled only in development + simulation."""
        return self.settings.is_development and self.simulation_mode

    # -- live video (optional, privacy-sensitive) --------------------------
    def live_video_enabled(self) -> bool:
        """Live camera feed is served only when explicitly enabled AND in
        development. The default posture is no live feed (see CLAUDE.md)."""
        return bool(self.settings.dashboard_live_video) and self.settings.is_development

    def video_token_required(self) -> bool:
        """True when a DASHBOARD_VIDEO_TOKEN is configured (feed is protected)."""
        return bool(self.settings.dashboard_video_token)

    def check_video_token(self, token: str | None) -> bool:
        """Constant-time check of a presented video token against the configured
        secret. Returns False if no token is configured or none was presented."""
        configured = self.settings.dashboard_video_token
        if not configured or not token:
            return False
        return hmac.compare_digest(str(token), configured)

    def _placeholder_image(self) -> Any:
        import numpy as np  # noqa: WPS433

        img = np.full((360, 640, 3), 32, dtype="uint8")
        try:
            import cv2  # noqa: WPS433

            msg = "SIMULATION - no live video" if self.simulation_mode else "Camera offline - no frames yet"
            cv2.putText(img, "VytalLink live feed", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 200, 255), 2)
            cv2.putText(img, msg, (24, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        except Exception:  # pragma: no cover - cv2 present on Jetson
            pass
        return img

    def _freshest_image(self) -> Any | None:
        """The freshest decoded camera frame (grabber rate), decoupled from the
        detection loop, for a low-latency raw live feed. Falls back to the last
        processed frame, then ``None``."""
        try:
            peek = self.camera.peek_latest()
        except Exception:  # pragma: no cover - defensive
            peek = None
        if peek is not None:
            return peek[0]
        return self._last_frame_image

    def _downscale_for_relay(self, img: Any) -> Any:
        """Downscale a COPY of ``img`` to fit RELAY_WIDTH×RELAY_HEIGHT (aspect
        preserved). Returns the original when downscaling is disabled (0) or the
        frame already fits. Never mutates the detection input."""
        rw, rh = self.settings.relay_width, self.settings.relay_height
        if rw <= 0 or rh <= 0:
            return img
        try:
            import cv2  # noqa: WPS433

            h, w = img.shape[:2]
            if w <= rw and h <= rh:
                return img
            scale = min(rw / float(w), rh / float(h))
            new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            return cv2.resize(img, new, interpolation=cv2.INTER_AREA)
        except Exception:  # pragma: no cover - defensive
            return img

    def _build_annotated_image(self) -> Any | None:
        """Draw the detector's EXISTING boxes for the last processed frame (never
        re-runs YOLO) plus detector FPS, frame age, and fall-state overlay."""
        base = self._last_frame_image
        if base is None:
            return self._freshest_image()
        try:
            import cv2  # noqa: WPS433

            img = base.copy()
            fall_set = self.settings.fall_class_set
            for d in self._last_detections:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                is_fall = d.class_name.lower() in fall_set
                color = (60, 60, 230) if is_fall else (90, 200, 120)  # BGR
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{d.class_name} {d.confidence:.2f}", (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            det_fps = getattr(self.detector, "inference_fps", lambda: None)()
            overlay = (f"state={self.state_machine.state.value}  det_fps={det_fps}  "
                       f"age={self._last_frame_age}s  score={self._last_evidence_score}")
            cv2.putText(img, overlay, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            return img
        except Exception:  # pragma: no cover - defensive
            return base

    def latest_frame_jpeg(
        self,
        quality: int | None = None,
        *,
        allow_placeholder: bool = True,
        annotated: bool | None = None,
    ) -> bytes | None:
        """Encode the live frame as JPEG bytes, downscaled to the relay size. No
        footage is written to disk. Call OFF the event loop (asyncio.to_thread).

        ``annotated`` (default: DASHBOARD_SHOW_DETECTIONS in a live mode) draws the
        detector's existing boxes — YOLO is never re-run. ``allow_placeholder=False``
        returns ``None`` when there is no real frame yet."""
        if annotated is None:
            annotated = self.settings.dashboard_show_detections and not self.simulation_mode
        img = self._build_annotated_image() if annotated else self._freshest_image()
        if img is None:
            if not allow_placeholder:
                return None
            img = self._placeholder_image()
        q = self.settings.relay_jpeg_quality if quality is None else int(quality)
        try:
            import cv2  # noqa: WPS433

            img = self._downscale_for_relay(img)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            return buf.tobytes() if ok else None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Live frame encode failed: %s", type(exc).__name__)
            return None

    def debug_metrics(self) -> dict[str, Any]:
        """Safe development metrics for diagnosing detections — no images,
        credentials, or filesystem paths."""
        ev = self.state_machine.current_event
        candidate_seconds = 0.0
        if ev is not None and self.state_machine.state == FallState.POSSIBLE_FALL:
            candidate_seconds = round(
                max(0.0, self.system_clock.monotonic() - ev.possible_since), 2
            )
        det = self._detector_health()
        cam = self.camera.health()
        return {
            "fall_state": self.state_machine.state.value,
            "frames_processed": self._frames_processed,
            "frames_with_fallen": self._frames_with_fallen,
            "frames_dropped_stale": self._frames_dropped_stale,
            "class_counts": dict(self._class_counts),
            "last_detections": self._last_detection_summary,
            "last_evidence": self._last_evidence,
            "evidence_score": self._last_evidence_score,
            "last_frame_age_seconds": self._last_frame_age,
            "fall_candidate_seconds": candidate_seconds,
            "confirm_seconds": self.settings.fall_confirm_seconds,
            "clear_seconds": self.settings.fall_clear_seconds,
            "confidence_threshold": self.settings.confidence_threshold,
            "require_transition": self.settings.require_fall_transition,
            "fall_classes": sorted(self.settings.fall_class_set),
            "transitions": list(self._transitions),
            "reconfirm_cooldown_seconds": self.settings.fall_reconfirm_cooldown_seconds,
            "rejections": det.get("rejection_counts"),
            "detector": {
                "device": det.get("device"),
                "device_label": det.get("device_label"),
                "inference_fps": det.get("inference_fps"),
                "avg_inference_ms": det.get("avg_inference_ms"),
                "inference_count": det.get("inference_count"),
                "min_fallen_box_area_frac": det.get("min_fallen_box_area_frac"),
                "reject_edge_clipped_fallen": det.get("reject_edge_clipped_fallen"),
            },
            "camera": {
                "effective_fps": cam.get("effective_fps"),
                "frames_grabbed": cam.get("frames_grabbed"),
                "frames_consumed": cam.get("frames_consumed"),
                "frames_dropped": cam.get("frames_dropped"),
                "frames_dropped_stale": self._frames_dropped_stale,
                "failed_reads": cam.get("failed_reads"),
                "reconnects": cam.get("reconnects"),
                "last_frame_age_seconds": cam.get("last_frame_age_seconds"),
            },
        }

    def _camera_status_value(self) -> str:
        if self.multi_camera_mode and self._multi_monitor is not None:
            return self._aggregate_camera_health(self._multi_monitor.health())["status"]
        return self.camera.status().value

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
            "camera_status": self._camera_status_value(),
            "detector": self._detector_health().get("name"),
            "wearable_status": self.wearable.status().value,
            "gpu_available": system_info.gpu_info().get("available", False),
            "simulation_active": self.simulation_mode,
            "controls_enabled": self.controls_enabled(),
            "last_update": isoformat(self.system_clock.now()),
        }


#: Severity ranking so multi-camera health can report the most-advanced state.
_FALL_STATE_RANK = {
    "normal": 0,
    "resolved": 1,
    "possible_fall": 2,
    "recovering": 3,
    "confirmed_fall": 4,
}


def _worst_fall_state(states: list[str]) -> str:
    if not states:
        return "normal"
    return max(states, key=lambda s: _FALL_STATE_RANK.get(s, 0))


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
