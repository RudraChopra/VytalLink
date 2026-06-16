"""Vision subsystem: camera providers and fall detectors."""

from vytallink.vision.base import CameraProvider
from vytallink.vision.detector_base import FallDetector, detections_to_evidence
from vytallink.vision.detector_simulated import Scenario, SimulatedFallDetector
from vytallink.vision.factory import build_camera, build_detector
from vytallink.vision.simulated import SimulatedCamera

__all__ = [
    "CameraProvider",
    "SimulatedCamera",
    "FallDetector",
    "SimulatedFallDetector",
    "Scenario",
    "detections_to_evidence",
    "build_camera",
    "build_detector",
]
