"""Centralized inference-device selection: CUDA → MPS → CPU.

Single source of truth so the detector, diagnostics, and health all agree on
which torch device runs inference. Selection is **probed** — each backend is
exercised with a real tensor operation, and any failure falls back safely to
the next option. A failed probe must never crash startup.

Selection order:

1. **CUDA**  — when ``torch.cuda.is_available()`` *and* a real CUDA tensor op
   succeeds (the Jetson path, once the CUDA wheel is installed).
2. **MPS**   — when CUDA is unavailable, Apple ``mps`` reports available, *and*
   a real MPS tensor op succeeds (the Apple-silicon dev path).
3. **CPU**   — otherwise.

Nothing here exposes absolute paths or host-private information; only the small
device strings (``"cuda:0"``, ``"mps"``, ``"cpu"``) and boolean availability
flags are surfaced.
"""

from __future__ import annotations

import functools
from typing import Any

from vytallink.common.logging_setup import get_logger

log = get_logger("common.device")

CUDA_DEVICE = "cuda:0"
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"


def _load_torch() -> Any | None:
    """Import torch, returning ``None`` (not raising) if it is unavailable."""
    try:
        import torch  # noqa: WPS433 (heavy, optional at import time)

        return torch
    except Exception as exc:  # pragma: no cover - torch present in this env
        log.warning("torch import failed (%s); inference device falls back to CPU", type(exc).__name__)
        return None


def _probe_cuda(torch: Any) -> bool:
    """True only if CUDA is available AND a real CUDA tensor op succeeds."""
    try:
        if not torch.cuda.is_available():
            return False
        x = torch.tensor([1.0, 1.0], device="cuda")
        ok = float((x + x).sum().item()) == 4.0
        torch.cuda.synchronize()
        return ok
    except Exception as exc:
        log.warning("CUDA probe failed (%s); not selecting CUDA", type(exc).__name__)
        return False


def _probe_mps(torch: Any) -> bool:
    """True only if MPS is available AND a real MPS tensor op succeeds."""
    try:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None or not backend.is_available():
            return False
        x = torch.tensor([1.0, 1.0], device="mps")
        ok = float((x + x).sum().item()) == 4.0
        synchronize_mps(torch)
        return ok
    except Exception as exc:
        log.warning("MPS probe failed (%s); not selecting MPS", type(exc).__name__)
        return False


def synchronize_mps(torch: Any) -> None:
    """Block until queued MPS work completes (needed for accurate timing)."""
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()
    except Exception:  # pragma: no cover - defensive
        pass


def synchronize_device(torch: Any, device: str | None) -> None:
    """Synchronize the active accelerator before timing. No-op for CPU."""
    try:
        if device and device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == MPS_DEVICE:
            synchronize_mps(torch)
    except Exception:  # pragma: no cover - defensive
        pass


def is_accelerator(device: str | None) -> bool:
    """True for a GPU-class device (CUDA or MPS), False for CPU/empty."""
    return bool(device) and (device.startswith("cuda") or device == MPS_DEVICE)


def device_label(device: str | None) -> str:
    """Human-readable name for an inference device string, for the dashboard.

    ``mps`` -> ``Apple MPS``, ``cuda[:N]`` -> ``CUDA``, ``cpu`` -> ``CPU``.
    """
    if not device:
        return "—"
    d = device.lower()
    if d == MPS_DEVICE:
        return "Apple MPS"
    if d.startswith("cuda"):
        return "CUDA"
    if d == CPU_DEVICE:
        return "CPU"
    return device


def select_device(preference: str | None = None) -> str:
    """Return the inference device string, probing in order CUDA → MPS → CPU.

    A non-empty ``preference`` is honored when its probe succeeds; ``"cpu"`` is
    always honored. A preference that fails its probe falls through to
    auto-selection rather than crashing. Never raises.
    """
    torch = _load_torch()
    if torch is None:
        return CPU_DEVICE

    if preference:
        pref = preference.strip().lower()
        if pref == CPU_DEVICE:
            return CPU_DEVICE
        if pref.startswith("cuda") and _probe_cuda(torch):
            return preference
        if pref == MPS_DEVICE and _probe_mps(torch):
            return MPS_DEVICE
        log.warning("Requested inference device %r is not usable; auto-selecting", preference)

    if _probe_cuda(torch):
        return CUDA_DEVICE
    if _probe_mps(torch):
        return MPS_DEVICE
    return CPU_DEVICE


@functools.lru_cache(maxsize=1)
def device_report() -> dict[str, Any]:
    """Safe, cached device snapshot for diagnostics / health.

    Contains only device strings, version strings, and boolean flags — no
    filesystem paths, hostnames, or other host-private information.
    """
    info: dict[str, Any] = {
        "selected_device": CPU_DEVICE,
        "cuda_available": False,
        "mps_available": False,
        "mps_built": False,
        "torch_version": None,
    }
    torch = _load_torch()
    if torch is None:
        return info

    info["torch_version"] = getattr(torch, "__version__", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is not None:
            info["mps_built"] = bool(getattr(backend, "is_built", lambda: False)())
            info["mps_available"] = bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive
        pass
    info["selected_device"] = select_device()
    return info


def reset_device_cache() -> None:
    """Clear the cached :func:`device_report` (used in tests)."""
    device_report.cache_clear()
