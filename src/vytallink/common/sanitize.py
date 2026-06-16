"""Secret-sanitization helpers.

Credentials must never appear in logs or API responses. In particular, RTSP
URLs frequently embed ``user:password@host`` — these must be redacted before
they are logged or surfaced.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

# Matches ``scheme://user:pass@host`` and similar credential-in-netloc forms.
_CREDENTIAL_NETLOC = re.compile(r"^(?P<userinfo>[^@/]+)@")

_REDACTED = "***REDACTED***"


def sanitize_url(url: str | None) -> str:
    """Redact any embedded credentials (``user:pass@``) from a URL.

    ``rtsp://alice:s3cret@cam.local:554/stream`` ->
    ``rtsp://***REDACTED***@cam.local:554/stream``

    Non-URL strings are returned unchanged (best-effort). Never raises.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.netloc and "@" in parts.netloc:
            host = parts.netloc.split("@", 1)[1]
            netloc = f"{_REDACTED}@{host}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return url
    except Exception:  # pragma: no cover - defensive: never let logging crash
        # Fall back to a crude redaction if parsing fails.
        return _CREDENTIAL_NETLOC.sub(f"{_REDACTED}@", url)


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
