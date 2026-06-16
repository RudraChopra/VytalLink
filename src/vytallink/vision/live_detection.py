"""Live detection diagnostic — ``python -m vytallink.vision.live_detection``.

Runs the **real** pipeline standalone (RTSP camera → YOLO detector → fall state
machine) so you can validate detections and tune thresholds without the web app.
It prints sanitized detections and performance metrics (camera FPS, inference
FPS/latency, device, dropped frames, reconnects, fall state). It does **not**
send caregiver alerts unless ``--alerts`` is passed, and it never saves footage
or opens a window.

This mirrors what ``MonitoringService`` does in live mode; the service remains
the production path (it persists events, dispatches alerts, and feeds the
dashboard, with camera+inference offloaded from the API event loop).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from vytallink.common.clock import SystemClock
from vytallink.config import DetectorMode, VisionMode, get_settings
from vytallink.events import FallEventStateMachine
from vytallink.vision import build_camera, build_detector, detections_to_evidence


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VytalLink live fall-detection diagnostic")
    p.add_argument("--seconds", type=float, default=60.0, help="duration; 0 = until Ctrl-C")
    p.add_argument("--alerts", action="store_true", help="actually emit console alerts (default: suppressed)")
    p.add_argument("--no-state", action="store_true", help="skip the state machine (raw detections only)")
    p.add_argument("--every", type=int, default=None, help="override PROCESS_EVERY_N_FRAMES")
    return p.parse_args(argv)


def _emit_console_alert(transition, sm) -> None:
    from vytallink.alerts.base import AlertEvent
    from vytallink.alerts.console import ConsoleAlertProvider

    ev = sm.current_event
    alert = AlertEvent(
        event_uid=transition.event_uid,
        timestamp=(ev.confirmed_time if ev else None) or transition.timestamp,
        confidence=ev.highest_confidence if ev else 0.0,
        source_device=ev.source_device if ev else "camera",
        state=transition.to_state.value,
        detection_count=ev.detection_count if ev else 0,
        simulated=False,
    )

    async def _go():
        provider = ConsoleAlertProvider()
        try:
            await provider.send(alert)
        finally:
            await provider.aclose()

    asyncio.run(_go())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = get_settings()
    # Force the real providers for this diagnostic, regardless of .env mode,
    # while keeping the .env camera/model/threshold values.
    settings = base.model_copy(
        update={"vision_mode": VisionMode.RTSP, "detector_mode": DetectorMode.YOLO}
    )
    every = max(1, args.every or settings.process_every_n_frames)

    print("VytalLink live detection diagnostic")
    print("=" * 60)
    if not settings.rtsp_url():
        print("[FAIL] No RTSP target configured (CAMERA_HOST / CAMERA_SOURCE).", file=sys.stderr)
        return 2

    clock = SystemClock()
    camera = build_camera(settings, clock=clock)
    detector = build_detector(settings, clock=clock)

    print(f"Camera   : {settings.sanitized_camera_source()}")
    print(f"Model    : {settings.model_path or 'models/fall_detection.pt (default)'}")
    print(f"Settings : imgsz={settings.image_size} conf={settings.confidence_threshold} "
          f"every={every} confirm={settings.fall_confirm_seconds}s clear={settings.fall_clear_seconds}s "
          f"transition_gate={settings.require_fall_transition}")
    print(f"Alerts   : {'ENABLED' if args.alerts else 'suppressed (diagnostic)'}")
    print("-" * 60)

    try:
        detector.load()
    except Exception as exc:  # noqa: BLE001 - sanitize
        print(f"[FAIL] detector load: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Detector : device={getattr(detector, 'device_str', '?')} "
          f"task={getattr(detector, 'task', '?')} classes={getattr(detector, 'class_names', {})}")

    try:
        camera.open()
    except Exception as exc:  # noqa: BLE001 - camera will retry with backoff
        print(f"(camera not open yet: {type(exc).__name__}; will retry) ")

    sm = None if args.no_state else FallEventStateMachine(
        confirm_seconds=settings.fall_confirm_seconds,
        clear_seconds=settings.fall_clear_seconds,
        cooldown_seconds=settings.alert_cooldown_seconds,
        source_device=settings.camera_device_id,
        clock=clock,
    )

    frames = 0
    processed = 0
    failed = 0
    frame_idx = 0
    start = time.monotonic()
    next_report = start + 2.0
    last_det_summary = "—"

    try:
        while args.seconds == 0 or (time.monotonic() - start) < args.seconds:
            frame = camera.read()
            if frame is None:
                failed += 1
                time.sleep(0.02)
                continue
            frames += 1
            frame_idx += 1
            if frame_idx % every != 0 or frame.image is None:
                continue
            processed += 1
            detections = detector.infer(frame)
            evidence, conf = detections_to_evidence(
                detections, settings.fall_class_set, settings.confidence_threshold
            )
            if detections:
                top = max(detections, key=lambda d: d.confidence)
                last_det_summary = f"{top.class_name}:{top.confidence:.2f} (n={len(detections)})"
            else:
                last_det_summary = "no detections"

            if sm is not None:
                for t in sm.observe(evidence, conf):
                    tag = "" if not t.alert else (" [ALERT]" if args.alerts else " [alert suppressed]")
                    print(f"  >> {t.from_state.value} -> {t.to_state.value} ({t.reason.value}){tag}")
                    if t.alert and args.alerts:
                        _emit_console_alert(t, sm)

            now = time.monotonic()
            if now >= next_report:
                ch = camera.health()
                state = sm.state.value if sm is not None else "n/a"
                print(
                    f"  t={now - start:5.1f}s cam_fps={ch.get('effective_fps')} "
                    f"inf_fps={detector.inference_fps()} inf_ms={detector.last_inference_ms} "
                    f"dev={detector.device_str} dropped={ch.get('frames_dropped')} "
                    f"reconnects={ch.get('reconnects')} state={state} det={last_det_summary}"
                )
                next_report = now + 2.0
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        camera.close()
        detector.close()

    print("-" * 60)
    print(f"frames received : {frames}")
    print(f"frames processed: {processed}")
    print(f"failed reads    : {failed}")
    print(f"avg inference ms: {round(detector.avg_inference_ms, 2) if detector.avg_inference_ms else None}")
    if frames == 0:
        print("\nRESULT: LIVE_FAIL (no frames — check camera).", file=sys.stderr)
        return 1
    print("\nRESULT: LIVE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
