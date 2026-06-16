"""Host metrics: CPU, memory, disk (with warning), and GPU availability.

All probes are best-effort and never raise — health reporting must not crash.
The GPU probe is cached (it may import torch, which is relatively heavy) so it
runs at most once per process.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from vytallink.common.logging_setup import get_logger

log = get_logger("monitoring.system_info")

try:  # psutil is available via system-site-packages
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil expected present
    psutil = None  # type: ignore


@lru_cache(maxsize=1)
def gpu_info() -> dict[str, Any]:
    """Detect GPU/CUDA availability for inference. Cached for the process.

    On this Jetson the system PyTorch is the CPU-only build, so CUDA reports
    unavailable until the Jetson CUDA wheel is installed (see hardware docs).
    """
    info: dict[str, Any] = {"available": False, "detail": "not probed", "framework": None}
    try:
        import torch  # noqa: WPS433 (heavy, hence cached)

        available = bool(torch.cuda.is_available())
        info.update(
            available=available,
            framework="torch",
            torch_version=getattr(torch, "__version__", "?"),
            cuda_build=getattr(getattr(torch, "version", None), "cuda", None),
            detail=(
                "CUDA available"
                if available
                else "torch present but CUDA unavailable (CPU-only build)"
            ),
        )
        if available:  # pragma: no cover - no CUDA torch in Phase 1 env
            info["device_count"] = torch.cuda.device_count()
    except Exception as exc:
        info["detail"] = f"torch not importable: {type(exc).__name__}"
    return info


def cpu_percent() -> float | None:
    if psutil is None:
        return None
    try:
        # Non-blocking; first call returns 0.0 then meaningful values thereafter.
        return round(psutil.cpu_percent(interval=None), 1)
    except Exception:  # pragma: no cover
        return None


def memory_info() -> dict[str, Any]:
    if psutil is None:
        return {"available": False}
    try:
        vm = psutil.virtual_memory()
        return {
            "available": True,
            "total_mb": round(vm.total / 1e6, 1),
            "used_mb": round(vm.used / 1e6, 1),
            "available_mb": round(vm.available / 1e6, 1),
            "percent": vm.percent,
        }
    except Exception:  # pragma: no cover
        return {"available": False}


def disk_info(path: str | Path, warning_percent: float = 90.0) -> dict[str, Any]:
    """Disk usage for the volume containing ``path``, with a warning flag."""
    try:
        target = Path(path)
        probe = target if target.exists() else target.parent
        usage = shutil.disk_usage(probe)
        percent = round(usage.used / usage.total * 100.0, 1) if usage.total else 0.0
        return {
            "total_gb": round(usage.total / 1e9, 2),
            "used_gb": round(usage.used / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
            "percent": percent,
            "warning": percent >= warning_percent,
            "warning_threshold": warning_percent,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"warning": False, "error": str(exc)}


def load_average() -> list[float] | None:
    try:
        import os

        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):  # pragma: no cover
        return None
