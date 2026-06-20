#!/usr/bin/env python3
"""Local iPhone-vitals relay SIMULATOR — POST synthetic vitals to a running server.

Sends SYNTHETIC, non-patient data to POST /api/vitals so you can exercise the
ingestion + compatibility-adapter + patient-state + alert-score path without a
real device. Defaults to localhost; sends one sample unless --count is raised.

NOTE: POST /api/vitals is a VytalLink-defined contract (no prior iPhone schema
existed). This validates THAT contract only — it does NOT prove the real phone
uses the same payload. Reconcile a sanitized real payload with
scripts/validate_vitals_fixture.py before relying on compatibility.

Each scenario declares its EXPECTED HTTP status; the simulator exits non-zero if
any send deviates, so it doubles as a quick contract check.

Examples:
  ./.venv/bin/python scripts/iphone_sim.py --scenario normal
  ./.venv/bin/python scripts/iphone_sim.py --scenario legacy_alias
  ./.venv/bin/python scripts/iphone_sim.py --all            # one of each, assert status
  ./.venv/bin/python scripts/iphone_sim.py --compare        # /latest vs /api/vitals/latest
  ./.venv/bin/python scripts/iphone_sim.py --count 5 --interval 2 --scenario normal
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

# (body_spec, expected_status). "_*" keys are simulator directives, not payload.
SCENARIOS: dict[str, tuple[dict, int]] = {
    "normal":          ({"heart_rate": 72, "respiratory_rate": 16, "posture": "upright", "motion": 0.1}, 200),
    "legacy_alias":    ({"hr": 73, "br": 15, "activity": 0.2, "recorded_at": "_now"}, 200),
    "high_hr":         ({"heart_rate": 185, "respiratory_rate": 22, "posture": "upright"}, 200),
    "low_hr":          ({"heart_rate": 35, "respiratory_rate": 12, "posture": "sitting"}, 200),
    "high_rr":         ({"heart_rate": 88, "respiratory_rate": 38, "posture": "upright"}, 200),
    "low_rr":          ({"heart_rate": 70, "respiratory_rate": 5, "posture": "lying"}, 200),
    "lying":           ({"heart_rate": 70, "respiratory_rate": 15, "posture": "lying"}, 200),
    "standing":        ({"heart_rate": 75, "respiratory_rate": 16, "posture": "standing"}, 200),
    "minimal":         ({"heart_rate": 72}, 200),
    "no_optional":     ({"heart_rate": 72}, 200),
    "slight_future":   ({"heart_rate": 72, "timestamp": "_future_ok"}, 200),   # within skew -> accepted
    "invalid_future":  ({"heart_rate": 72, "timestamp": "_future_bad"}, 400),  # far future -> rejected
    "stale":           ({"heart_rate": 72, "timestamp": "_stale"}, 200),       # old but within reject -> accepted, classified stale
    "too_old":         ({"heart_rate": 72, "timestamp": "_too_old"}, 400),     # beyond reject window
    "duplicate":       ({"heart_rate": 72, "sample_id": "sim-dup-1"}, 200),    # second send -> idempotent
    "conflict":        ({"heart_rate": 72, "hr": 80}, 422),                    # conflicting aliases
    "invalid_string":  ({"heart_rate": "abc"}, 422),
    "out_of_range":    ({"heart_rate": 500}, 422),
    "nan":             ({"_raw": "{\"heart_rate\": 1e400}"}, None),            # 4xx (400 or 422)
    "malformed":       ({"_raw": "{not valid json"}, 422),
}


def _iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def build(spec: dict):
    spec = dict(spec)
    if "_raw" in spec:
        return None, spec["_raw"].encode()
    ts = spec.get("timestamp")
    repl = {"_now": 0, "_future_ok": 30, "_future_bad": 7200, "_stale": -600, "_too_old": -7200}
    for k, v in list(spec.items()):
        if v == "_now" or v in repl:
            spec[k] = _iso(repl.get(v, 0))
    spec.setdefault("device_id", "sim-iphone")
    return spec, None


def post(url: str, body, raw):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:  # noqa: BLE001
        return None, f"(error: {type(e).__name__})"


def status_ok(code, expected) -> bool:
    if expected is None:
        return code is not None and 400 <= code < 500   # any 4xx
    return code == expected


def run_scenario(base: str, name: str) -> bool:
    spec, expected = SCENARIOS[name]
    body, raw = build(spec)
    code, text = post(base.rstrip("/") + "/api/vitals", body, raw)
    try:
        summary = json.loads(text)
        if isinstance(summary, dict):
            summary = {k: summary[k] for k in ("accepted", "idempotent", "contract_form", "accepted_fields", "detail") if k in summary}
    except Exception:
        summary = str(text)[:120]
    ok = status_ok(code, expected)
    print(f"  {name:<14} HTTP {code} {'OK ' if ok else 'UNEXPECTED'} (want {expected}): {summary}")
    return ok


def compare(base: str) -> bool:
    a, _ = post(base.rstrip("/") + "/api/vitals", *build(SCENARIOS["normal"][0]))  # ensure a sample
    import urllib.request as u
    try:
        latest = json.loads(u.urlopen(base.rstrip("/") + "/latest", timeout=8).read())
        canon = json.loads(u.urlopen(base.rstrip("/") + "/api/vitals/latest", timeout=8).read())
    except Exception as e:  # noqa: BLE001
        print(f"  compare: fetch error {type(e).__name__}"); return False
    # The two share the same service; legacy + new keys must be present in both.
    keys = {"vital", "simulated", "vision", "freshness", "alert"}
    ok = keys <= set(latest) and keys <= set(canon)
    print(f"  /latest and /api/vitals/latest both expose {sorted(keys)}: {ok}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Local iPhone vitals simulator (synthetic data only).")
    ap.add_argument("--url", default="http://127.0.0.1:5050")
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument("--all", action="store_true", help="run every scenario once, assert expected status")
    ap.add_argument("--compare", action="store_true", help="compare /latest vs /api/vitals/latest")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    print(f"iPhone simulator -> {args.url}  (synthetic data only)")
    failures = 0
    if args.compare:
        if not compare(args.url):
            failures += 1
    elif args.all:
        for name in SCENARIOS:
            if name == "duplicate":
                run_scenario(args.url, "duplicate")  # first send
            if not run_scenario(args.url, name):
                failures += 1
    else:
        import time
        for i in range(max(1, args.count)):
            if not run_scenario(args.url, args.scenario):
                failures += 1
            if args.count > 1 and i < args.count - 1:
                time.sleep(args.interval if 0 < args.interval <= 300 else 2.0)
    print(f"done: {failures} unexpected result(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
