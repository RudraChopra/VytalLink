"""Self-diagnostics for VytalLink (used by scripts/diagnose.sh).

Runs a series of checks and prints a PASS/WARN/FAIL/SKIP report. Exits non-zero
only if a hard check FAILs (WARN does not fail the run). Importable so it can be
unit-tested.
"""

from __future__ import annotations

import platform
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from vytallink.common.errors import ConfigError
from vytallink.config import DetectorMode, Settings, VisionMode, get_settings

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _port_free(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_host = "0.0.0.0" if host in ("0.0.0.0", "") else host
    try:
        s.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def run_diagnostics(settings: Settings) -> list[Check]:
    checks: list[Check] = []

    # 1. Environment
    py = sys.version_info
    checks.append(
        Check(
            "environment",
            PASS if py >= (3, 10) else WARN,
            f"Python {py.major}.{py.minor}.{py.micro} on {platform.machine()} / {platform.system()}",
        )
    )

    # 2. Virtual environment
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checks.append(
        Check("python_env", PASS if in_venv else WARN,
              "running inside project venv" if in_venv else "not in a venv (run scripts/setup.sh)")
    )

    # 3. Imports
    missing = []
    for mod in ("fastapi", "uvicorn", "pydantic", "httpx", "vytallink.api.server",
                "vytallink.monitoring.service"):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{mod} ({type(exc).__name__})")
    checks.append(
        Check("imports", PASS if not missing else FAIL,
              "all core imports OK" if not missing else "missing: " + ", ".join(missing))
    )

    # 4. Configuration
    checks.append(
        Check("configuration", PASS,
              f"env={settings.env.value} vision={settings.vision_mode.value} "
              f"detector={settings.detector_mode.value} port={settings.port}")
    )

    # 5. Database access
    try:
        from vytallink.database import Database

        db = Database(settings.database_path)
        version = db.initialize()
        health = db.health()
        db.close()
        ok = bool(health.get("ok"))
        checks.append(
            Check("database", PASS if ok else FAIL,
                  f"schema v{version} at {settings.database_path} (writable={health.get('writable')})")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("database", FAIL, f"{type(exc).__name__}: {exc}"))

    # 6. Port availability
    free = _port_free(settings.host, settings.port)
    checks.append(
        Check("port", PASS if free else WARN,
              f"{settings.host}:{settings.port} is free" if free
              else f"{settings.host}:{settings.port} in use (app already running?)")
    )

    # 7. GPU visibility
    try:
        from vytallink.monitoring import system_info

        gpu = system_info.gpu_info()
        if gpu.get("available"):
            checks.append(Check("gpu", PASS, gpu.get("detail", "CUDA available")))
        else:
            checks.append(
                Check("gpu", WARN,
                      f"{gpu.get('detail')} — not required for Phase 1 simulation; "
                      "install the Jetson CUDA PyTorch wheel before real inference")
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("gpu", WARN, f"probe failed: {exc}"))

    # 8. Camera configuration presence
    if settings.vision_mode == VisionMode.SIMULATION:
        checks.append(Check("camera_config", PASS, "simulation (no camera required)"))
    elif settings.camera_source:
        checks.append(Check("camera_config", PASS,
                            f"{settings.vision_mode.value} source configured (redacted)"))
    else:
        checks.append(Check("camera_config", WARN if not settings.is_production else FAIL,
                            f"VISION_MODE={settings.vision_mode.value} but CAMERA_SOURCE is empty"))

    # 9. Model configuration presence
    if settings.detector_mode == DetectorMode.SIMULATION:
        checks.append(Check("model_config", PASS, "simulation detector (no model required)"))
    elif settings.model_path and Path(settings.model_path).expanduser().exists():
        checks.append(Check("model_config", PASS, f"model present at {settings.model_path}"))
    else:
        checks.append(Check("model_config", WARN if not settings.is_production else FAIL,
                            f"DETECTOR_MODE={settings.detector_mode.value} but MODEL_PATH missing/not found"))

    # 10. Wearable mode
    checks.append(Check("wearable", PASS, f"mode={settings.wearable_mode.value}"))

    # 11. Disk space
    try:
        from vytallink.monitoring import system_info

        disk = system_info.disk_info(settings.database_path, settings.disk_warning_percent)
        checks.append(
            Check("disk", WARN if disk.get("warning") else PASS,
                  f"{disk.get('free_gb')} GB free ({disk.get('percent')}% used)")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("disk", WARN, f"probe failed: {exc}"))

    return checks


def overall_status(checks: list[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    if any(c.status == WARN for c in checks):
        return WARN
    return PASS


def main() -> int:
    print("VytalLink diagnostics")
    print("=" * 60)
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[FAIL] configuration: {exc}")
        return 1
    checks = run_diagnostics(settings)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:>4}] {c.name.ljust(width)}  {c.detail}")
    overall = overall_status(checks)
    print("=" * 60)
    print(f"Overall: {overall}")
    return 1 if overall == FAIL else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
