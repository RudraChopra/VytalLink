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

import threading
from typing import Any

from vytallink.common.logging_setup import get_logger

log = get_logger("common.device")

CUDA_DEVICE = "cuda:0"
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"

# --- one-time accelerator probing (Apple-silicon MPS thread-safety) ----------
# Probing an accelerator means creating a real tensor on it and synchronizing —
# i.e. running Metal/CUDA command-buffer work. On Apple MPS, doing that on a
# thread OTHER than the one running model inference aborts the process with a
# Metal "addScheduledHandler after commit" assertion (it is a C++ assert -> not
# catchable in Python). The detector resolves its device on the dedicated
# inference thread during startup; the only other caller was `device_report()`
# (reached from /health on the event-loop thread), which created a SECOND MPS
# tensor and raced steady-state inference.
#
# Fix: the actual probe runs at most ONCE per process (memoized under a lock),
# so it lands on the inference thread that resolves first; and `device_report()`
# never probes — it uses only static availability checks plus the device the
# detector publishes here. Together these guarantee no accelerator command
# buffer is ever created off the inference thread.
_probe_lock = threading.Lock()
_cuda_usable: bool | None = None
_mps_usable: bool | None = None
_resolved_device: str | None = None  # published by select_device() after it resolves


def _load_torch() -> Any | None:
    """Import torch, returning ``None`` (not raising) if it is unavailable."""
    try:
        import torch  # noqa: WPS433 (heavy, optional at import time)

        return torch
    except Exception as exc:  # pragma: no cover - torch present in this env
        log.warning("torch import failed (%s); inference device falls back to CPU", type(exc).__name__)
        return None


def _run_cuda_probe(torch: Any) -> bool:
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


def _run_mps_probe(torch: Any) -> bool:
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


def _probe_cuda(torch: Any) -> bool:
    """True only if CUDA is available AND a real CUDA tensor op succeeds.

    Memoized: the tensor op runs at most once per process, so it executes on the
    first caller's thread (the inference thread during startup) and never again.
    """
    global _cuda_usable
    with _probe_lock:
        if _cuda_usable is None:
            _cuda_usable = _run_cuda_probe(torch)
        return _cuda_usable


def _probe_mps(torch: Any) -> bool:
    """True only if MPS is available AND a real MPS tensor op succeeds.

    Memoized (see :func:`_probe_cuda`) so the MPS command-buffer op runs exactly
    once, on the inference thread — never concurrently from another thread.
    """
    global _mps_usable
    with _probe_lock:
        if _mps_usable is None:
            _mps_usable = _run_mps_probe(torch)
        return _mps_usable


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


def _publish_resolved_device(dev: str) -> str:
    """Record the device the inference path resolved, so probe-free callers
    (``device_report`` / health) report the authoritative value."""
    global _resolved_device
    _resolved_device = dev
    return dev


def resolved_device() -> str | None:
    """The device the inference path last resolved, or ``None`` before startup."""
    return _resolved_device


def select_device(preference: str | None = None) -> str:
    """Return the inference device string, probing in order CUDA → MPS → CPU.

    A non-empty ``preference`` is honored when its probe succeeds; ``"cpu"`` is
    always honored. A preference that fails its probe falls through to
    auto-selection rather than crashing. Never raises. The chosen device is
    published for probe-free callers (see :func:`device_report`).

    The probe (a real accelerator tensor op) runs at most once per process, so
    calling this from the dedicated inference thread at startup is what fixes the
    Apple-MPS cross-thread abort — no other thread ever creates a probe tensor.
    """
    torch = _load_torch()
    if torch is None:
        return _publish_resolved_device(CPU_DEVICE)

    if preference:
        pref = preference.strip().lower()
        if pref == CPU_DEVICE:
            return _publish_resolved_device(CPU_DEVICE)
        if pref.startswith("cuda") and _probe_cuda(torch):
            return _publish_resolved_device(preference)
        if pref == MPS_DEVICE and _probe_mps(torch):
            return _publish_resolved_device(MPS_DEVICE)
        log.warning("Requested inference device %r is not usable; auto-selecting", preference)

    if _probe_cuda(torch):
        return _publish_resolved_device(CUDA_DEVICE)
    if _probe_mps(torch):
        return _publish_resolved_device(MPS_DEVICE)
    return _publish_resolved_device(CPU_DEVICE)


def _static_device_guess(cuda_available: bool, mps_available: bool) -> str:
    """Probe-free best guess at the inference device from availability flags
    alone (no tensor op). Only used before the inference path has resolved."""
    if cuda_available:
        return CUDA_DEVICE
    if mps_available:
        return MPS_DEVICE
    return CPU_DEVICE


def device_report() -> dict[str, Any]:
    """Safe device snapshot for diagnostics / health.

    Uses ONLY static availability checks (``is_available``/``is_built`` create no
    command buffer) plus the device the inference path published — it never runs
    an accelerator probe, so it is safe to call from the event-loop / any thread
    concurrently with inference. Contains only device strings, version strings,
    and boolean flags — no filesystem paths or host-private information.
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
    # Authoritative once the inference path has resolved; a probe-free guess
    # before that. NEVER calls select_device() (which would probe on this thread).
    info["selected_device"] = _resolved_device or _static_device_guess(
        info["cuda_available"], info["mps_available"]
    )
    return info


def reset_device_cache() -> None:
    """Reset memoized probe results + the published device (used in tests)."""
    global _cuda_usable, _mps_usable, _resolved_device
    with _probe_lock:
        _cuda_usable = None
        _mps_usable = None
        _resolved_device = None
