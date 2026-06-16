"""VytalLink application entrypoint.

Configures logging, builds the FastAPI app (which owns the MonitoringService
lifecycle), and runs it under uvicorn. uvicorn installs SIGINT/SIGTERM handlers
and triggers the lifespan shutdown, giving us graceful shutdown of the
background loops and providers.

Run:
    python -m vytallink.app
"""

from __future__ import annotations

import sys

from vytallink import __phase__, __version__
from vytallink.common.errors import ConfigError
from vytallink.common.logging_setup import configure_logging, get_logger
from vytallink.config import get_settings


def build_app():
    """Build the configured FastAPI app (used by uvicorn and tests)."""
    from vytallink.api.server import create_app

    settings = get_settings()
    return create_app(settings)


def main() -> int:
    try:
        settings = get_settings()
    except ConfigError as exc:
        # Clear, actionable startup error — no stack trace, no secrets.
        print(f"[VytalLink] Configuration error:\n{exc}", file=sys.stderr)
        return 2

    configure_logging(settings.log_level, settings.log_dir)
    log = get_logger("app")
    log.info("Starting %s %s (%s)", "VytalLink", __version__, __phase__)
    log.info("Configuration: %s", settings.safe_summary())
    log.info(
        "Dashboard will listen on http://%s:%d (LAN-accessible if host is 0.0.0.0)",
        settings.host,
        settings.port,
    )

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a hard dep
        print("[VytalLink] uvicorn is not installed. Run scripts/setup.sh.", file=sys.stderr)
        return 2

    from vytallink.api.server import create_app

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
