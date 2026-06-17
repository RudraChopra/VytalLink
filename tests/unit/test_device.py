"""Tests for centralized inference-device selection (CUDA → MPS → CPU).

Hardware-independent: a tiny fake ``torch`` module drives the real probe and
selection logic, so these pass on a CPU-only box, a CUDA Jetson, or an Apple
MPS Mac without any actual accelerator.
"""

from __future__ import annotations

import types

import pytest

from vytallink.common import device as dev
from vytallink.common.device import (
    CPU_DEVICE,
    CUDA_DEVICE,
    MPS_DEVICE,
    select_device,
)


# --- a minimal fake torch -------------------------------------------------
class _FakeTensor:
    def __init__(self, data):
        self.data = list(data)

    def __add__(self, other):
        return _FakeTensor([a + b for a, b in zip(self.data, other.data)])

    def sum(self):
        return types.SimpleNamespace(item=lambda: sum(self.data))


def _fake_torch(*, cuda_avail, cuda_raises=False, mps_avail=False, mps_raises=False):
    def tensor(data, device=None):
        if device == "cuda" and cuda_raises:
            raise RuntimeError("CUDA error: device-side assert triggered")
        if device == "mps" and mps_raises:
            raise NotImplementedError("aten::some_op not implemented for MPS")
        return _FakeTensor(data)

    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_avail,
        synchronize=lambda: None,
        device_count=lambda: 1,
    )
    mps_backend = types.SimpleNamespace(
        is_available=lambda: mps_avail,
        is_built=lambda: mps_avail,
    )
    return types.SimpleNamespace(
        __version__="fake-2.12",
        cuda=cuda,
        backends=types.SimpleNamespace(mps=mps_backend),
        mps=types.SimpleNamespace(synchronize=lambda: None),
        tensor=tensor,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    dev.reset_device_cache()
    yield
    dev.reset_device_cache()


def _use(monkeypatch, torch):
    monkeypatch.setattr(dev, "_load_torch", lambda: torch)


# --- selection order ------------------------------------------------------
def test_cuda_is_preferred_when_available(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=True, mps_avail=True))
    assert select_device() == CUDA_DEVICE


def test_mps_selected_when_cuda_unavailable(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=True))
    assert select_device() == MPS_DEVICE


def test_cpu_fallback_when_no_accelerator(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=False))
    assert select_device() == CPU_DEVICE


def test_cpu_when_torch_missing(monkeypatch):
    monkeypatch.setattr(dev, "_load_torch", lambda: None)
    assert select_device() == CPU_DEVICE


# --- probe failures must fall back, not crash -----------------------------
def test_cuda_probe_failure_falls_back_to_mps(monkeypatch):
    # CUDA *reports* available but the real tensor op raises -> skip CUDA.
    _use(monkeypatch, _fake_torch(cuda_avail=True, cuda_raises=True, mps_avail=True))
    assert select_device() == MPS_DEVICE


def test_cuda_probe_failure_falls_back_to_cpu(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=True, cuda_raises=True, mps_avail=False))
    assert select_device() == CPU_DEVICE


def test_mps_probe_failure_falls_back_to_cpu(monkeypatch):
    # MPS reports available but the real op raises (unsupported) -> CPU.
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=True, mps_raises=True))
    assert select_device() == CPU_DEVICE


def test_probe_functions_return_bool_on_failure(monkeypatch):
    assert dev._probe_cuda(_fake_torch(cuda_avail=True, cuda_raises=True)) is False
    assert dev._probe_cuda(_fake_torch(cuda_avail=False)) is False
    assert dev._probe_cuda(_fake_torch(cuda_avail=True)) is True
    assert dev._probe_mps(_fake_torch(cuda_avail=False, mps_avail=True, mps_raises=True)) is False
    assert dev._probe_mps(_fake_torch(cuda_avail=False, mps_avail=True)) is True


# --- explicit preference --------------------------------------------------
def test_explicit_cpu_preference_is_honored(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=True, mps_avail=True))
    assert select_device("cpu") == CPU_DEVICE


def test_unusable_preference_falls_through_to_autoselect(monkeypatch):
    # Prefer CUDA but it isn't usable -> auto-select finds MPS.
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=True))
    assert select_device("cuda:0") == MPS_DEVICE


def test_mps_preference_honored_when_usable(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=True))
    assert select_device("mps") == MPS_DEVICE


# --- device_report is safe + reflects selection ---------------------------
def test_device_report_is_safe_and_consistent(monkeypatch):
    _use(monkeypatch, _fake_torch(cuda_avail=False, mps_avail=True))
    rpt = dev.device_report()
    assert rpt["selected_device"] == MPS_DEVICE
    assert rpt["mps_available"] is True
    assert rpt["cuda_available"] is False
    assert rpt["torch_version"] == "fake-2.12"
    # No filesystem paths / host-private info anywhere in the report.
    blob = str(rpt)
    assert "/Users/" not in blob and "/home/" not in blob
