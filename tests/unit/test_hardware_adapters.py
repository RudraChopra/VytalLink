"""Tests for the (dormant) hardware adapters: safe behavior without hardware.

These verify the RTSP/file/YOLO adapters fail cleanly and never leak
credentials — without requiring a real camera, model, or GPU.
"""

from __future__ import annotations

import pytest

from vytallink.common.errors import CameraError, DetectorError
from vytallink.config import load_settings
from vytallink.vision.factory import build_camera, build_detector
from vytallink.vision.file_source import VideoFileCamera
from vytallink.vision.rtsp import RTSPCamera


def test_rtsp_safe_source_redacts_credentials():
    cam = RTSPCamera("rtsp://alice:s3cret@cam.local:554/Streaming/Channels/101")
    assert "alice" not in cam.safe_source
    assert "s3cret" not in cam.safe_source
    assert "cam.local:554" in cam.safe_source


def test_rtsp_empty_connection_fails_cleanly():
    cam = RTSPCamera("")
    with pytest.raises(CameraError):
        cam.open()


def test_video_file_missing_raises_camera_error():
    cam = VideoFileCamera("/no/such/file.mp4")
    with pytest.raises(CameraError) as exc:
        cam.open()
    assert "not found" in str(exc.value).lower()


def test_yolo_missing_model_raises_clear_error():
    from vytallink.vision.detector_yolo import YoloFallDetector

    det = YoloFallDetector("/no/such/model.pt")
    assert det.loaded is False
    with pytest.raises(DetectorError) as exc:
        det.load()
    assert "MODEL_PATH" in str(exc.value)


def test_factory_builds_rtsp_camera_with_safe_source():
    settings = load_settings(
        vision_mode="rtsp",
        camera_source="rtsp://cam.local:554/s",
        camera_username="bob",
        camera_password="pw",
    )
    cam = build_camera(settings)
    assert isinstance(cam, RTSPCamera)
    assert "bob" not in cam.safe_source and "pw" not in cam.safe_source


def test_factory_builds_yolo_detector_unloaded():
    settings = load_settings(detector_mode="yolo", model_path="/tmp/none.pt")
    det = build_detector(settings)
    assert det.name == "yolo"
    assert det.loaded is False  # not loaded until load() with a real model


def test_factory_tensorrt_not_implemented_in_phase1():
    settings = load_settings(detector_mode="tensorrt")
    with pytest.raises(NotImplementedError):
        build_detector(settings)
