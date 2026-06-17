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
    """Detect the inference accelerator (CUDA, Apple MPS, or CPU). Cached.

    ``available`` reports CUDA specifically (back-compat with the Jetson path);
    ``selected_device`` reports what inference actually runs on after probing
    CUDA → MPS → CPU (see :mod:`vytallink.common.device`). On the Jetson the
    system PyTorch is CPU-only until the CUDA wheel is installed; on Apple
    silicon MPS is selected. Only device strings / flags are surfaced — no
    paths or host-private info.
    """
    from vytallink.common.device import device_report

    info: dict[str, Any] = {"available": False, "detail": "not probed", "framework": None}
    try:
        import torch  # noqa: WPS433 (heavy, hence cached)

        rpt = device_report()
        cuda = rpt["cuda_available"]
        mps = rpt["mps_available"]
        selected = rpt["selected_device"]
        if cuda:
            detail = "CUDA available"
        elif mps:
            detail = "Apple MPS available (CUDA unavailable)"
        else:
            detail = "torch present but no GPU accelerator (using CPU)"
        info.update(
            available=cuda,
            framework="torch",
            torch_version=getattr(torch, "__version__", "?"),
            cuda_build=getattr(getattr(torch, "version", None), "cuda", None),
            cuda_available=cuda,
            mps_available=mps,
            mps_built=rpt["mps_built"],
            selected_device=selected,
            detail=detail,
        )
        if cuda:
            info["device_count"] = torch.cuda.device_count()
            try:
                info["device_name"] = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                info["compute_capability"] = f"{cap[0]}.{cap[1]}"
            except Exception:  # pragma: no cover - defensive
                pass
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
