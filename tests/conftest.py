"""Shared pytest fixtures for VytalLink tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vytallink.common.clock import ManualClock
from vytallink.database import Database, Repositories


@pytest.fixture
def manual_clock() -> ManualClock:
    """A deterministic clock for sleep-free timing tests."""
    return ManualClock()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_vytallink.db"


@pytest.fixture
def database(temp_db_path: Path, manual_clock: ManualClock) -> Database:
    db = Database(temp_db_path, clock=manual_clock)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)
