#!/usr/bin/env python3
"""Local iPhone-vitals relay SIMULATOR — POST sample vitals to a running server.

Sends SYNTHETIC, non-patient data to POST /api/vitals so you can exercise the
ingestion + patient-state + alert-score path without a real device. Defaults to
localhost and sends a single sample unless --count is raised.

NOTE: POST /api/vitals is a VytalLink-defined contract (no prior iPhone schema
existed). Verify the path/payload against the real phone before relying on it.

Examples:
  ./.venv/bin/python scripts/iphone_sim.py                       # one normal sample
  ./.venv/bin/python scripts/iphone_sim.py --scenario high_hr
  ./.venv/bin/python scripts/iphone_sim.py --scenario stale
  ./.venv/bin/python scripts/iphone_sim.py --count 5 --interval 2
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

# Synthetic scenarios (no real patient information).
SCENARIOS = {
    "normal":    {"heart_rate": 72, "respiratory_rate": 16, "posture": "upright", "motion": 0.1},
    "high_hr":   {"heart_rate": 185, "respiratory_rate": 22, "posture": "upright", "motion": 0.4},
    "low_hr":    {"heart_rate": 35, "respiratory_rate": 12, "posture": "sitting", "motion": 0.0},
    "resp":      {"heart_rate": 88, "respiratory_rate": 38, "posture": "upright", "motion": 0.2},
    "lying":     {"heart_rate": 70, "respiratory_rate": 15, "posture": "lying", "motion": 0.0},
    "minimal":   {"heart_rate": 72},
    "stale":     {"heart_rate": 72, "_stale": True},       # timestamp ~10 min old -> rejected
    "duplicate": {"heart_rate": 72, "sample_id": "sim-dup-1"},
    "malformed": {"_malformed": True},                      # not valid JSON -> 4xx
}


def _now_iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def _post(url: str, body, *, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:  # noqa: BLE001
        return None, f"(error: {type(e).__name__})"


def build_body(scenario: str):
    spec = dict(SCENARIOS[scenario])
    if spec.pop("_malformed", False):
        return None, b"{not valid json"
    if spec.pop("_stale", False):
        spec["timestamp"] = _now_iso(-600)  # 10 minutes old
    else:
        spec["timestamp"] = _now_iso()
    spec.setdefault("device_id", "sim-iphone")
    return spec, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Local iPhone vitals simulator (synthetic data only).")
    ap.add_argument("--url", default="http://127.0.0.1:5050", help="server base URL")
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument("--count", type=int, default=1, help="number of samples to send")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between samples (count>1)")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/api/vitals"
    print(f"POST {endpoint}  scenario={args.scenario}  count={args.count}  (synthetic data)")
    for i in range(args.count):
        body, raw = build_body(args.scenario)
        code, text = _post(endpoint, body, raw=raw)
        try:
            summary = json.loads(text)
        except Exception:
            summary = text[:160]
        print(f"  [{i+1}/{args.count}] HTTP {code}: {summary}")
        if args.count > 1 and i < args.count - 1:
            import time
            if not (300 >= args.interval > 0):  # guard pathological intervals
                args.interval = 2.0
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
