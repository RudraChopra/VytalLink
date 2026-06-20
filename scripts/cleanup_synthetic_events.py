#!/usr/bin/env python3
"""List and (with --confirm) delete SYNTHETIC fall events from the LOCAL database.

Synthetic events are created while synthetic fall testing is active and are tagged
event_type='fall_synthetic'. This tool ONLY ever touches rows with that exact
marker — real events (event_type='fall') can never be deleted by it.

Safety:
  * Dry-run by default (lists only); deletion requires --confirm.
  * Refuses to run against a production environment unless
    ALLOW_SYNTHETIC_CLEANUP_IN_PROD=1 is explicitly set.
  * Operates only on the configured local database (VYTALLINK_DATABASE_PATH / default).
  * Never runs automatically — invoke it by hand.

Usage:
  ./.venv/bin/python scripts/cleanup_synthetic_events.py            # list
  ./.venv/bin/python scripts/cleanup_synthetic_events.py --confirm  # delete
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="List/delete synthetic (validation) fall events.")
    ap.add_argument("--confirm", action="store_true", help="actually delete (default: list only)")
    args = ap.parse_args()

    from vytallink.config import get_settings
    from vytallink.database.db import Database
    from vytallink.database.maintenance import (
        count_real_events,
        delete_synthetic_events,
        list_synthetic_events,
    )

    settings = get_settings()
    if settings.is_production and os.environ.get("ALLOW_SYNTHETIC_CLEANUP_IN_PROD") != "1":
        print(
            "Refusing to run against a production environment. "
            "Set ALLOW_SYNTHETIC_CLEANUP_IN_PROD=1 to override.",
            file=sys.stderr,
        )
        return 2

    db = Database(settings.database_path)
    db.initialize()
    try:
        synthetic = list_synthetic_events(db)
        real = count_real_events(db)
        print(f"Database: {os.path.basename(settings.database_path)}  (real events kept: {real})")
        if not synthetic:
            print("No synthetic (fall_synthetic) events found.")
            return 0
        print(f"Synthetic events ({len(synthetic)}):")
        for e in synthetic:
            print(
                f"  {e['event_uid'][:12]}  state={e['state']:<14} "
                f"src={e['source_device']:<10} conf={e['highest_confidence']}"
            )
        if not args.confirm:
            print(
                "\nDry run. Re-run with --confirm to DELETE the events above. "
                "Real events are never touched."
            )
            return 0
        n = delete_synthetic_events(db)
        print(f"\nDeleted {n} synthetic event(s). Real events untouched ({count_real_events(db)} remain).")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
