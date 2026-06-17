"""Factories that build the configured camera and detector providers."""

from __future__ import annotations

from vytallink.common.clock import Clock
from vytallink.common.logging_setup import get_logger
from vytallink.config import DetectorMode, Settings, VisionMode
from vytallink.vision.base import CameraProvider
from vytallink.vision.detector_base import FallDetector
from vytallink.vision.detector_simulated import SimulatedFallDetector
from vytallink.vision.simulated import SimulatedCamera

log = get_logger("vision.factory")


def build_camera(settings: Settings, clock: Clock | None = None) -> CameraProvider:
    """Construct the camera provider for the configured VISION_MODE."""
    sid = settings.camera_device_id
    if settings.vision_mode == VisionMode.SIMULATION:
        return SimulatedCamera(source_id=sid, clock=clock, stale_timeout=settings.wearable_sample_seconds * 3)
    if settings.vision_mode == VisionMode.FILE:
        from vytallink.vision.file_source import VideoFileCamera

        return VideoFileCamera(settings.camera_source, source_id=sid, clock=clock)
    if settings.vision_mode == VisionMode.RTSP:
        from vytallink.vision.rtsp import RTSPCamera

        return RTSPCamera(settings.camera_connection_string(), source_id=sid, clock=clock)
    if settings.vision_mode == VisionMode.HTTP_MJPEG:
        from vytallink.vision.http_source import HttpCamera

        return HttpCamera(
            stream_url=settings.camera_http_stream_url,
            snapshot_url=settings.camera_http_snapshot_url,
            bearer_token=settings.camera_http_bearer_token,
            source_id=sid,
            clock=clock,
        )
    raise ValueError(f"Unknown vision mode: {settings.vision_mode}")  # pragma: no cover


def build_detector(settings: Settings, clock: Clock | None = None) -> FallDetector:
    """Construct the fall detector for the configured DETECTOR_MODE."""
    if settings.detector_mode == DetectorMode.SIMULATION:
        return SimulatedFallDetector()
    if settings.detector_mode == DetectorMode.YOLO:
        from vytallink.vision.detector_yolo import YoloFallDetector

        return YoloFallDetector(
            settings.model_path,
            image_size=settings.image_size,
            confidence=settings.confidence_threshold,
            require_transition=settings.require_fall_transition,
            clock=clock,
        )
    if settings.detector_mode == DetectorMode.TENSORRT:
        # TensorRT export/engine path is intentionally not implemented until the
        # real model loads and ordinary GPU inference is confirmed.
        raise NotImplementedError(
            "DETECTOR_MODE=tensorrt is not enabled in Phase 1. Validate the YOLO "
            "model on GPU first, then export an engine. See docs/hardware_needed.md."
        )
    raise ValueError(f"Unknown detector mode: {settings.detector_mode}")  # pragma: no cover
