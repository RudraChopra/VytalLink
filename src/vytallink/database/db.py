"""Thread-safe SQLite connection manager with migrations and health checks."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from vytallink.common.clock import Clock, SystemClock
from vytallink.common.errors import DatabaseError
from vytallink.common.logging_setup import get_logger
from vytallink.database.schema import LATEST_SCHEMA_VERSION, MIGRATIONS

log = get_logger("database")


class Database:
    """A single SQLite connection guarded by a re-entrant lock.

    SQLite handles our Phase 1 scale comfortably. We use a single shared
    connection with ``check_same_thread=False`` and serialize all access with
    an ``RLock`` so it is safe to use from the asyncio loop and any worker
    threads. WAL journaling improves read/write concurrency; the ``-wal`` and
    ``-shm`` side files are gitignored.
    """

    def __init__(self, path: str | Path, clock: Clock | None = None) -> None:
        self.path = Path(path)
        self.clock: Clock = clock or SystemClock()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            if self.path.parent and str(self.path) != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                conn = sqlite3.connect(
                    str(self.path), check_same_thread=False, timeout=30.0
                )
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                raise DatabaseError(f"Could not open database {self.path}: {exc}") from exc
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL is unsupported for in-memory DBs; guard it.
            if str(self.path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
            return conn

    def initialize(self) -> int:
        """Create/upgrade the schema. Returns the schema version after migration."""
        conn = self.connect()
        with self._lock:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > LATEST_SCHEMA_VERSION:
                raise DatabaseError(
                    f"Database schema version {current} is newer than this build "
                    f"supports ({LATEST_SCHEMA_VERSION}). Upgrade the application."
                )
            applied = 0
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                log.info("Applying database migration to version %d", version)
                try:
                    conn.execute("BEGIN")
                    for stmt in statements:
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {version}")
                    conn.commit()
                    applied += 1
                except sqlite3.Error as exc:
                    conn.rollback()
                    raise DatabaseError(
                        f"Migration to version {version} failed: {exc}"
                    ) from exc
            if applied:
                log.info("Database migrations complete (now at version %d)", LATEST_SCHEMA_VERSION)
            return conn.execute("PRAGMA user_version").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None

    # -- query helpers (all parameterized) ---------------------------------
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> sqlite3.Cursor:
        conn = self.connect()
        with self._lock:
            try:
                cur = conn.execute(sql, tuple(params or ()))
                conn.commit()
                return cur
            except sqlite3.Error as exc:
                conn.rollback()
                raise DatabaseError(f"Query failed: {exc}") from exc

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        conn = self.connect()
        with self._lock:
            try:
                cur = conn.executemany(sql, [tuple(p) for p in seq])
                conn.commit()
                return cur
            except sqlite3.Error as exc:
                conn.rollback()
                raise DatabaseError(f"Bulk query failed: {exc}") from exc

    def query_one(self, sql: str, params: Sequence[Any] | None = None) -> sqlite3.Row | None:
        conn = self.connect()
        with self._lock:
            cur = conn.execute(sql, tuple(params or ()))
            return cur.fetchone()

    def query_all(self, sql: str, params: Sequence[Any] | None = None) -> list[sqlite3.Row]:
        conn = self.connect()
        with self._lock:
            cur = conn.execute(sql, tuple(params or ()))
            return cur.fetchall()

    # -- health ------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Lightweight health probe. Never raises."""
        try:
            conn = self.connect()
            with self._lock:
                conn.execute("SELECT 1").fetchone()
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            return {
                "ok": True,
                "schema_version": int(version),
                "path": str(self.path),
                "writable": self._is_writable(),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": str(exc), "path": str(self.path)}

    def _is_writable(self) -> bool:
        if str(self.path) == ":memory:":
            return True
        target = self.path if self.path.exists() else self.path.parent
        import os

        return os.access(target, os.W_OK)

    def now_iso(self) -> str:
        from vytallink.common.clock import isoformat

        return isoformat(self.clock.now())  # type: ignore[return-value]
