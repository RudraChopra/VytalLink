"""Multi-camera configuration: parse ``CAMERA_{N}_*`` env vars into configs.

Distinct from the single-camera ``CAMERA_*`` settings (which stay for the
existing simulation/relay/single-RTSP paths). Multi-camera mode is **off by
default** — committed defaults define no ``CAMERA_{N}_*`` keys, so
:func:`cameras_from_env` returns an empty list and the app stays single-camera.

Credentials are kept on the :class:`CameraConfig` and only ever assembled into a
URL in memory; nothing here logs or returns a credential-bearing URL — use
:meth:`CameraConfig.safe_label` for any human-facing output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import quote

_INDEX_RE = re.compile(r"^CAMERA_(\d+)_HOST$")
_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CameraConfig:
    """One configured camera. ``camera_id`` is a stable, credential-free label."""

    camera_id: str
    host: str
    port: int = 554
    stream_path: str = "stream1"
    username: str = ""
    password: str = ""
    enabled: bool = True

    def rtsp_url(self) -> str:
        """Assemble the credential-bearing RTSP URL (creds URL-encoded so an
        email username / special-char password stays well-formed). **Never log
        this** — use :meth:`safe_label`."""
        creds = ""
        if self.username:
            user = quote(self.username, safe="")
            creds = f"{user}:{quote(self.password, safe='')}@" if self.password else f"{user}@"
        path = self.stream_path.strip()
        if path and not path.startswith("/"):
            path = "/" + path
        port = f":{self.port}" if self.port else ""
        return f"rtsp://{creds}{self.host}{port}{path}"

    def safe_label(self) -> str:
        """Credential-free identifier for logs/health: ``camera_1 (host:port)``."""
        return f"{self.camera_id} ({self.host}:{self.port})"


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def cameras_from_env(
    environ: Mapping[str, str], *, require_enabled: bool = True
) -> list[CameraConfig]:
    """Parse ``CAMERA_{N}_HOST`` (and siblings) into a list of configs.

    A camera index ``N`` is *configured* when ``CAMERA_{N}_HOST`` is non-empty.
    It is *enabled* when ``CAMERA_{N}_ENABLED`` is truthy (``1/true/yes/on``).
    With ``require_enabled`` (the default) only enabled cameras are returned, so
    the app activates multi-camera mode only when explicitly switched on.
    Indices are returned in ascending numeric order.
    """
    indices: list[int] = []
    for key in environ:
        m = _INDEX_RE.match(key)
        if m and str(environ.get(key, "")).strip():
            indices.append(int(m.group(1)))
    out: list[CameraConfig] = []
    for i in sorted(set(indices)):
        host = str(environ.get(f"CAMERA_{i}_HOST", "")).strip()
        if not host:
            continue
        enabled = str(environ.get(f"CAMERA_{i}_ENABLED", "")).strip().lower() in _TRUE
        if require_enabled and not enabled:
            continue
        out.append(
            CameraConfig(
                camera_id=f"camera_{i}",
                host=host,
                port=_to_int(environ.get(f"CAMERA_{i}_PORT"), 554),
                stream_path=str(environ.get(f"CAMERA_{i}_STREAM_PATH", "stream1")).strip() or "stream1",
                username=str(environ.get(f"CAMERA_{i}_USERNAME", "")),
                password=str(environ.get(f"CAMERA_{i}_PASSWORD", "")),
                enabled=enabled,
            )
        )
    return out
