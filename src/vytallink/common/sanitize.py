"""Secret-sanitization helpers.

Credentials must never appear in logs or API responses. In particular, RTSP
URLs frequently embed ``user:password@host`` — these must be redacted before
they are logged or surfaced.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# Matches ``scheme://user:pass@host`` and similar credential-in-netloc forms.
_CREDENTIAL_NETLOC = re.compile(r"^(?P<userinfo>[^@/]+)@")

_REDACTED = "***REDACTED***"


def sanitize_url(url: str | None) -> str:
    """Redact any embedded credentials (``user:pass@``) from a URL.

    ``rtsp://alice:s3cret@cam.local:554/stream`` ->
    ``rtsp://***REDACTED***@cam.local:554/stream``

    Robust against passwords that contain ``@`` (uses the parsed host/port,
    which split on the *last* ``@``, rather than string-splitting on the first)
    and against scheme-less URLs (``user:pass@host/path``). Non-URL strings are
    returned unchanged (best-effort). Never raises.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.netloc and "@" in parts.netloc:
            # parts.hostname/port use rpartition('@'), so a '@' in the password
            # cannot leak the host. Rebuild the netloc from the parsed host.
            host = parts.hostname or ""
            if ":" in host:  # IPv6 literal
                host = f"[{host}]"
            port = f":{parts.port}" if parts.port else ""
            netloc = f"{_REDACTED}@{host}{port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        if not parts.netloc and "@" in url:
            # Scheme-less authority, e.g. "user:p@ss@host:554/path" — urlsplit
            # parsed the whole thing as a path. Redact the userinfo portion.
            authority, sep, rest = url.partition("/")
            if "@" in authority:
                hostpart = authority.rpartition("@")[2]
                return f"{_REDACTED}@{hostpart}{sep}{rest}"
        return url
    except Exception:  # pragma: no cover - defensive: never let logging crash
        # Fall back to a crude redaction if parsing fails.
        return _CREDENTIAL_NETLOC.sub(f"{_REDACTED}@", url)


def safe_path(path: str | Path | None) -> str:
    """Reduce a filesystem path to a credential-safe identifier.

    Absolute paths frequently embed the local username (``/Users/alice/...``)
    and other host details that must never appear in public API responses or
    logs. We surface only the *basename* — enough to identify the file
    (``api_test.db``, ``vytallink.db``) without leaking the home directory or
    user. The in-memory sentinel and empty values pass through unchanged.
    """
    if not path:
        return ""
    text = str(path)
    if text == ":memory:":
        return text
    return Path(text).name or text


def sanitize_secret(value: str | None) -> str:
    """Render a secret as a fixed redaction token if present, else empty."""
    return _REDACTED if value else ""


def mask_value(value: str | None, keep: int = 0) -> str:
    """Mask a value, optionally revealing the last ``keep`` characters.

    Useful for log lines like ``webhook secret set (…cret)`` without exposing
    the secret. ``keep=0`` fully masks.
    """
    if not value:
        return ""
    if keep <= 0 or len(value) <= keep:
        return _REDACTED
    return f"…{value[-keep:]}"
