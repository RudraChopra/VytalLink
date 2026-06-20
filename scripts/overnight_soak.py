"""Overnight soak: collect comprehensive metrics AND post safe synthetic iPhone
vitals on a schedule, to a running two-camera app. Detects memory/queue growth,
reconnects, model reloads, duplicate snapshots, and vitals freshness transitions.

Safe-by-design: synthetic vitals only (never real patient data), no synthetic
falls generated here, /health is credential-free (scanned anyway). Writes
diagnostics/overnight_soak.jsonl + a begin/mid/end summary. Does NOT claim
long-term stability — reports only what was observed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

PORT = 5050
BASE = f"http://127.0.0.1:{PORT}"


def _get(path: str):
    t0 = time.monotonic()
    try:
        body = urllib.request.urlopen(BASE + path, timeout=8).read().decode()
        return json.loads(body), round((time.monotonic() - t0) * 1000, 1), body
    except Exception as e:  # noqa: BLE001
        return {"_error": type(e).__name__}, None, ""


def _post_vitals(spec: dict):
    try:
        data = json.dumps(spec).encode()
        req = urllib.request.Request(BASE + "/api/vitals", data=data,
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.getcode(), json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001
        return None, {}


def _rss_cpu(pid: str):
    try:
        out = subprocess.run(["ps", "-p", pid, "-o", "rss=,%cpu="], capture_output=True, text=True).stdout.split()
        return (round(int(out[0]) / 1024, 1), float(out[1])) if len(out) >= 2 else (None, None)
    except Exception:  # noqa: BLE001
        return (None, None)


def _now_iso(offset=0.0):
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 7200.0
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    os.makedirs("diagnostics", exist_ok=True)
    out_path = "diagnostics/overnight_soak.jsonl"

    try:
        from vytallink.config.settings import get_settings
        secrets = {x for c in get_settings().configured_cameras() for x in (c.username, c.password) if x}
        pid = open("run/vytallink.pid").read().strip()
    except Exception:  # noqa: BLE001
        secrets, pid = set(), ""

    accepted = rejected = duplicate = 0
    samples = []
    end = time.monotonic() + duration
    tick = 0
    with open(out_path, "w") as fh:
        while time.monotonic() < end:
            tick += 1
            # --- vitals posting schedule (safe synthetic patterns) ---
            # most ticks: normal; every 5th: elevated; every 7th: SKIP (let vitals
            # age toward stale); every 11th: duplicate retry; some: missing optional.
            posted = None
            if tick % 7 == 0:
                posted = "skip"   # pause -> exercise aging/stale
            elif tick % 5 == 0:
                code, _ = _post_vitals({"heart_rate": 150, "respiratory_rate": 22, "posture": "upright",
                                        "timestamp": _now_iso(), "device_id": "soak-iphone"})
                posted = ("high_hr", code)
            elif tick % 11 == 0:
                code, body = _post_vitals({"heart_rate": 72, "sample_id": "soak-dup",
                                           "timestamp": _now_iso(), "device_id": "soak-iphone"})
                posted = ("dup", code, body.get("idempotent"))
                if body.get("idempotent"):
                    duplicate += 1
            else:
                spec = {"heart_rate": 70, "timestamp": _now_iso(), "device_id": "soak-iphone"}
                if tick % 3:
                    spec.update({"respiratory_rate": 15, "posture": "upright"})
                code, _ = _post_vitals(spec)
                posted = ("normal", code)
            if isinstance(posted, tuple) and posted[1] == 200:
                accepted += 1
            elif isinstance(posted, tuple) and posted[1] not in (200, None):
                rejected += 1

            # --- metrics ---
            h, h_ms, h_body = _get("/health")
            p, p_ms, _ = _get("/api/patient")
            _l, l_ms, _ = _get("/latest")
            rss, cpu = _rss_cpu(pid)
            v = h.get("vision", {}) if "_error" not in h else {}
            cams = v.get("cameras", {})
            leak = sum(h_body.count(s) for s in secrets) if h_body else 0
            samples.append({
                "t": round(time.monotonic(), 1),
                "overall": h.get("overall"), "model_state": h.get("model", {}).get("state"),
                "model_load_count": v.get("model_load_count"), "queue_peak": v.get("inference_queue_peak"),
                "rss_mb": rss, "cpu": cpu, "secret_leak": leak,
                "snap_written": h.get("persistence", {}).get("snapshots_written"),
                "snap_failures": h.get("persistence", {}).get("snapshot_failures"),
                "snap_total": h.get("persistence", {}).get("incident_snapshots_total"),
                "alerts": h.get("alerts", {}).get("status"),
                "patient_version": p.get("version"), "vitals_freshness": p.get("freshness", {}).get("vitals"),
                "alert_level": p.get("alert", {}).get("level"),
                "lat_ms": {"health": h_ms, "patient": p_ms, "latest": l_ms},
                "cameras": {c: {k: cm.get(k) for k in ("connected", "fps", "reconnects", "failed_reads",
                            "dropped_frames", "fall_state", "tick_errors", "alive", "last_frame_age_ms")}
                            for c, cm in cams.items()},
            })
            fh.write(json.dumps(samples[-1]) + "\n"); fh.flush()
            time.sleep(interval)

    if not samples:
        print("SOAK: no samples"); return 1
    first, mid, last = samples[0], samples[len(samples) // 2], samples[-1]
    rss = [s["rss_mb"] for s in samples if s["rss_mb"] is not None]
    cam_ids = sorted(last["cameras"].keys())
    summary = {
        "samples": len(samples), "duration_s": round(last["t"] - first["t"], 1),
        "rss_begin_mb": first["rss_mb"], "rss_mid_mb": mid["rss_mb"], "rss_end_mb": last["rss_mb"],
        "rss_delta_mb": round(last["rss_mb"] - first["rss_mb"], 1) if rss else None,
        "rss_peak_mb": max(rss) if rss else None,
        "cpu_range": [min(s["cpu"] for s in samples if s["cpu"] is not None),
                      max(s["cpu"] for s in samples if s["cpu"] is not None)] if any(s["cpu"] for s in samples) else None,
        "model_load_counts_seen": sorted({s["model_load_count"] for s in samples if s["model_load_count"] is not None}),
        "queue_peak": max((s["queue_peak"] or 0) for s in samples),
        "overall_states_seen": sorted({s["overall"] for s in samples if s["overall"]}),
        "vitals_freshness_seen": sorted({s["vitals_freshness"] for s in samples if s["vitals_freshness"]}),
        "alert_levels_seen": sorted({s["alert_level"] for s in samples if s["alert_level"]}),
        "vitals_accepted": accepted, "vitals_rejected": rejected, "vitals_duplicate_suppressed": duplicate,
        "snapshots_written_end": last["snap_written"], "snapshot_failures_end": last["snap_failures"],
        "snapshot_total_end": last["snap_total"],
        "reconnect_max_per_cam": {c: max((s["cameras"].get(c, {}).get("reconnects") or 0) for s in samples) for c in cam_ids},
        "tick_errors_max_per_cam": {c: max((s["cameras"].get(c, {}).get("tick_errors") or 0) for s in samples) for c in cam_ids},
        "workers_alive_throughout": all(all(s["cameras"].get(c, {}).get("alive") for c in cam_ids) for s in samples),
        "secret_leaks_total": sum(s["secret_leak"] or 0 for s in samples),
        "lat_health_max_ms": max((s["lat_ms"]["health"] or 0) for s in samples),
        "lat_patient_max_ms": max((s["lat_ms"]["patient"] or 0) for s in samples),
    }
    with open("diagnostics/overnight_soak_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
