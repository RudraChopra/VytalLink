"""FastAPI application: health, status, events, devices, vitals, simulation.

The app holds a :class:`MonitoringService` on ``app.state.service``. A lifespan
context starts/stops the service (and its background loops) with the app, so
the API stays responsive while monitoring runs. Errors are returned as JSON
without leaking stack traces or secrets.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vytallink import APP_NAME, __version__
from vytallink.api.schemas import (
    LabelRequest,
    ResolveRequest,
    VitalsIngest,
    device_to_dict,
    event_to_dict,
    incident_vital_to_dict,
    vital_to_dict,
)
from vytallink.common.errors import NotFoundError, VytalLinkError
from vytallink.common.logging_setup import get_logger
from vytallink.config import Settings, get_settings
from vytallink.monitoring import MonitoringService

log = get_logger("api")

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"
_STATIC_DIR = _DASHBOARD_DIR / "static"


def create_app(
    settings: Settings | None = None, service: MonitoringService | None = None
) -> FastAPI:
    settings = settings or get_settings()
    service = service or MonitoringService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The app owns the service lifecycle: start background loops on startup,
        # stop them cleanly on shutdown. Idempotent start/stop make this safe
        # even if a caller passed an already-constructed service.
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title=f"{APP_NAME} API",
        version=__version__,
        description="VytalLink Phase 1 monitoring API (simulation-first).",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = settings

    _register_error_handlers(app)
    _register_routes(app)
    _mount_dashboard(app)
    return app


def _svc(request: Request) -> MonitoringService:
    return request.app.state.service


def _video_authorized(request: Request, svc: MonitoringService) -> bool:
    """Authorize a live-video request. When DASHBOARD_VIDEO_TOKEN is unset the
    feed is open (flag-gated only); when set, an ``Authorization: Bearer <token>``
    header is required. The token is never read from the URL and never logged."""
    if not svc.video_token_required():
        return True
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and svc.check_video_token(token.strip())


def _video_unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": "unauthorized", "detail": "Valid video token required."},
        headers={"WWW-Authenticate": "Bearer"},
    )


# --- error handling -------------------------------------------------------
def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})

    @app.exception_handler(VytalLinkError)
    async def _domain_error(request: Request, exc: VytalLinkError):
        return JSONResponse(status_code=400, content={"error": "error", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Never leak stack traces or secrets to clients.
        log.exception("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An internal error occurred."},
        )


# --- routes ---------------------------------------------------------------
def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return _svc(request).health()

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        return _svc(request).status()

    @app.get("/api/events")
    async def list_events(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        state: str | None = Query(None),
    ) -> dict[str, Any]:
        repos = _svc(request).repos
        items = repos.events.list(limit=limit, offset=offset, state=state)
        return {
            "items": [event_to_dict(e) for e in items],
            "limit": limit,
            "offset": offset,
            "returned": len(items),
            "total": repos.events.count(state=state),
        }

    @app.get("/api/events/{event_id}")
    async def get_event(request: Request, event_id: str) -> dict[str, Any]:
        repos = _svc(request).repos
        ev = repos.events.require(event_id)
        alerts = repos.alerts.list_for_event(event_id)
        data = event_to_dict(ev, alerts)
        # Attach the incident vitals snapshot (if one was captured at confirmation).
        snap = repos.incident_vitals.get_by_event(event_id)
        data["incident_vitals"] = incident_vital_to_dict(snap) if snap else None
        return data

    @app.post("/api/events/{event_id}/label")
    async def label_event(request: Request, event_id: str, body: LabelRequest) -> dict[str, Any]:
        row = await _svc(request).label_event(event_id, body.label.value)
        return event_to_dict(row)

    @app.post("/api/events/{event_id}/resolve")
    async def resolve_event(
        request: Request, event_id: str, body: ResolveRequest | None = None
    ) -> dict[str, Any]:
        note = body.note if body else None
        row = await _svc(request).resolve_event(event_id, note)
        return event_to_dict(row)

    @app.get("/api/devices")
    async def list_devices(request: Request) -> dict[str, Any]:
        devices = _svc(request).repos.devices.list()
        return {"items": [device_to_dict(d) for d in devices], "returned": len(devices)}

    def _latest_vital_payload(svc: MonitoringService) -> dict[str, Any]:
        """Single source of truth for the latest-vital response (shared by the
        canonical route and the legacy ``/latest`` alias). Legacy top-level
        fields (``vital``, ``simulated``) are preserved; structured
        ``vision``/``freshness``/``alert`` sections are added for new clients."""
        v = svc.repos.vitals.latest()
        ps = svc.patient_state()
        return {
            "vital": vital_to_dict(v) if v is not None else None,
            "simulated": (v.simulated if v is not None else svc.simulation_mode),
            "vision": ps["vision"],
            "freshness": ps["freshness"],
            "alert": ps["alert"],
        }

    @app.get("/api/vitals/latest")
    async def latest_vital(request: Request) -> dict[str, Any]:
        # Canonical latest-vital endpoint.
        return _latest_vital_payload(_svc(request))

    @app.get("/latest")
    async def latest_vital_alias(request: Request) -> dict[str, Any]:
        # Backward-compatible alias for the legacy iPhone vitals relay / prior
        # VytalLink, which polled GET /latest. Identical response, schema, and
        # no-vitals behavior as /api/vitals/latest (the canonical route).
        return _latest_vital_payload(_svc(request))

    @app.get("/api/vitals")
    async def list_vitals(
        request: Request,
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        device_id: str | None = Query(None),
    ) -> dict[str, Any]:
        repos = _svc(request).repos
        items = repos.vitals.list(limit=limit, offset=offset, device_id=device_id)
        return {
            "items": [vital_to_dict(v) for v in items],
            "limit": limit,
            "offset": offset,
            "returned": len(items),
            "total": repos.vitals.count(),
            "simulated": _svc(request).simulation_mode,
        }

    @app.post("/api/vitals")
    async def ingest_vitals(request: Request, body: VitalsIngest) -> dict[str, Any]:
        # iPhone vitals ingestion. NOTE: this path/payload is a VytalLink-defined
        # contract (no prior iPhone schema existed) — verify against the device.
        svc = _svc(request)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > svc.settings.vitals_ingest_max_bytes:
            raise HTTPException(status_code=413, detail="payload too large")
        # ValueError (bad/future/old timestamp) -> 400 via the error handler; a
        # malformed/invalid body is rejected by pydantic with 422. Neither leaks
        # internals, and the service never logs values or the full payload.
        row, idempotent = svc.ingest_vitals(body)
        md = row.metadata or {}
        return {
            "accepted": True,
            "idempotent": idempotent,
            "device_id": row.device_id,
            "source_timestamp": row.timestamp,
            "received_at": md.get("received_at"),
            "contract_form": md.get("contract_form"),
            "accepted_fields": sorted(body.accepted_aliases.keys()),
        }

    @app.get("/api/patient")
    async def patient_state(request: Request) -> dict[str, Any]:
        # Full normalized patient state (vitals + per-camera vision + freshness +
        # informational alert score). The alert score is NOT a medical diagnosis.
        return _svc(request).patient_state()

    # -- live camera feed (opt-in, development only; never saves footage) ---
    @app.get("/api/camera/snapshot.jpg")
    async def camera_snapshot(request: Request):
        svc = _svc(request)
        if not svc.live_video_enabled():
            return JSONResponse(
                status_code=404, content={"error": "not_found", "detail": "Live video is disabled."}
            )
        if not _video_authorized(request, svc):
            return _video_unauthorized()
        jpeg = await asyncio.to_thread(svc.latest_frame_jpeg)
        if jpeg is None:
            return JSONResponse(
                status_code=503, content={"error": "unavailable", "detail": "No frame available."}
            )
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/stream")
    async def camera_stream(request: Request):
        svc = _svc(request)
        if not svc.live_video_enabled():
            return JSONResponse(
                status_code=404, content={"error": "not_found", "detail": "Live video is disabled."}
            )
        if not _video_authorized(request, svc):
            return _video_unauthorized()

        frame_interval = 1.0 / max(0.1, svc.settings.relay_max_fps)

        async def gen():
            # Encode each frame OFF the event loop and cap the rate (RELAY_MAX_FPS)
            # so the live feed never starves the API. Stops when the client
            # disconnects; an encode error is logged (sanitized) and closes cleanly.
            while True:
                if await request.is_disconnected():
                    break
                try:
                    jpeg = await asyncio.to_thread(svc.latest_frame_jpeg)
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("Live MJPEG frame failed: %s", type(exc).__name__)
                    break
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(frame_interval)

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    # -- detector debug (development only; no images/credentials/paths) -----
    @app.get("/api/detector/debug")
    async def detector_debug(request: Request):
        svc = _svc(request)
        if not svc.settings.is_development:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "detail": "Debug metrics are development-only."},
            )
        return svc.debug_metrics()

    # -- simulation controls (development + simulation only) ---------------
    @app.post("/api/simulation/fall")
    async def simulate_fall(request: Request):
        svc = _svc(request)
        if not svc.controls_enabled():
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden", "detail": "Simulation controls are disabled."},
            )
        return await svc.simulate_fall()

    @app.post("/api/simulation/normal")
    async def simulate_normal(request: Request):
        svc = _svc(request)
        if not svc.controls_enabled():
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden", "detail": "Simulation controls are disabled."},
            )
        return await svc.simulate_normal()

    @app.post("/api/simulation/reset")
    async def simulate_reset(request: Request):
        svc = _svc(request)
        if not svc.controls_enabled():
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden", "detail": "Simulation controls are disabled."},
            )
        return await svc.simulate_reset()


# --- dashboard mounting ---------------------------------------------------
def _mount_dashboard(app: FastAPI) -> None:
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    index = _TEMPLATES_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse(
            f"<h1>{APP_NAME}</h1><p>API is running. Dashboard template not found.</p>"
        )
