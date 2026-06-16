"""Centralized logging configuration with rotating file handlers.

Provides:

* A console handler (stderr) for interactive runs.
* A rotating file handler writing to ``logs/vytallink.log`` (5 MB x 5 files).

Secrets must never be logged; callers are responsible for sanitizing values
(see :mod:`vytallink.common.sanitize`). This module only wires up handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(
    level: str = "INFO",
    log_dir: Path | str | None = None,
    log_file: str = "vytallink.log",
    *,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging once. Safe to call repeatedly (idempotent).

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        log_dir: Directory for the rotating log file. If ``None``, only the
            console handler is installed (useful for tests).
        log_file: Base file name for the rotating log.
        max_bytes: Rotation threshold per file.
        backup_count: Number of rotated files to retain.
        force: Reconfigure even if already configured.

    Returns:
        The root ``vytallink`` logger.
    """
    global _CONFIGURED
    root = logging.getLogger("vytallink")

    if _CONFIGURED and not force:
        return root

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric_level)

    # Clear existing handlers when forcing, to avoid duplicate lines.
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(numeric_level)
        root.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        target = log_path / log_file
        has_file = any(
            isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
        )
        if not has_file:
            file_handler = logging.handlers.RotatingFileHandler(
                target, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            root.addHandler(file_handler)

    # Don't propagate to the python root logger (avoids duplicate output).
    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``vytallink`` namespace."""
    if name.startswith("vytallink"):
        return logging.getLogger(name)
    return logging.getLogger(f"vytallink.{name}")
