#!/usr/bin/env python3
"""Validate a SANITIZED iPhone vitals JSON fixture against the current contract.

This is the privacy-preserving alternative to a live capture endpoint: the
developer captures ONE real payload, removes personal identifiers/PHI, saves it
as JSON, and runs this to see whether POST /api/vitals would accept it and how
its fields map to the canonical model. It prints field NAMES, TYPES, contract
form, and size — never the values — so its output is safe to share. It is a
manual developer command (no server endpoint, nothing persisted, off by default).

Real-device reconciliation workflow:
  1. Capture one real payload from the phone (developer, with consent).
  2. Remove names, identifiers, and any PHI you do not need.
  3. Save it as a .json file.
  4. Run: ./.venv/bin/python scripts/validate_vitals_fixture.py payload.json
  5. If fields are unmapped, add aliases to api/schemas.VITALS_ALIASES deliberately.
  6. Add the sanitized fixture under tests/ and re-run the suite.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a sanitized vitals fixture (types/keys only, no values).")
    ap.add_argument("fixture", help="path to a SANITIZED JSON payload")
    ap.add_argument("--show-keys", action="store_true", help="also print top-level key NAMES (no values)")
    args = ap.parse_args()

    try:
        with open(args.fixture, "rb") as fh:
            raw = fh.read()
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot read/parse fixture: {type(exc).__name__}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("fixture must be a JSON object", file=sys.stderr)
        return 2

    from vytallink.api.schemas import VITALS_ALIASES, VitalsIngest

    known = {a for keys in VITALS_ALIASES.values() for a in keys}
    present = list(data.keys())
    unmapped = [k for k in present if k not in known]
    print(f"fixture size: {len(raw)} bytes; top-level keys: {len(present)}")
    print("field types:", {k: type(v).__name__ for k, v in data.items()})  # types, not values
    if args.show_keys:
        print("keys:", present)
    print("unmapped keys (ignored by the adapter):", unmapped or "none")

    try:
        model = VitalsIngest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        print("REJECTED:", str(exc).splitlines()[0][:200])
        return 1
    print("ACCEPTED. contract_form:", model.contract_form)
    print("canonical fields populated:", sorted(model.accepted_aliases.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
