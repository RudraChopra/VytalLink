"""Two-camera soak monitor: sample /health every INTERVAL for DURATION, record
per-camera + process metrics to a gitignored JSONL, and print a begin/mid/end
summary with growth/anomaly detection. Saves no media; redacts nothing because
/health is already credential-free (it is scanned here to be sure).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

PORT = 5050
BASE = f"http://127.0.0.1:{PORT}"


def rss_cpu(pid: str):
    try:
        out = subprocess.run(["ps", "-p", pid, "-o", "rss=,%cpu="],
                             capture_output=True, text=True).stdout.split()
        return (round(int(out[0]) / 1024, 1), float(out[1])) if len(out) >= 2 else (None, None)
    except Exception:
        return (None, None)


def health():
    try:
        return json.loads(urllib.request.urlopen(BASE + "/health", timeout=8).read())
    except Exception as e:
        return {"_error": type(e).__name__}


def sample(pid, secrets):
    h = health()
    blob = json.dumps(h)
    leak = sum(blob.count(s) for s in secrets)
    v = h.get("vision", {}) if "_error" not in h else {}
    cams = v.get("cameras", {})
    rss, cpu = rss_cpu(pid)
    return {
        "t": round(time.monotonic(), 1),
        "overall": h.get("overall"),
        "model_state": h.get("model", {}).get("state"),
        "model_load_count": v.get("model_load_count"),
        "queue_depth": v.get("inference_queue_depth"),
        "queue_peak": v.get("inference_queue_peak"),
        "rss_mb": rss, "cpu": cpu, "secret_leak": leak,
        "cameras": {cid: {k: c.get(k) for k in ("connected", "fps", "last_frame_age_ms",
                    "reconnects", "dropped_frames", "failed_reads", "backlog",
                    "confirmed_falls", "tick_errors", "alive")} for cid, c in cams.items()},
    }


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    out_path = "diagnostics/soak.jsonl"

    try:
        from vytallink.config.settings import get_settings
        s = get_settings()
        secrets = {x for c in s.configured_cameras() for x in (c.username, c.password) if x}
        pid = open("run/vytallink.pid").read().strip()
    except Exception:
        secrets, pid = set(), ""

    import os
    os.makedirs("diagnostics", exist_ok=True)  # gitignored output dir
    samples = []
    end = time.monotonic() + duration
    with open(out_path, "w") as f:
        while time.monotonic() < end:
            s = sample(pid, secrets)
            samples.append(s)
            f.write(json.dumps(s) + "\n"); f.flush()
            time.sleep(interval)

    if not samples:
        print("SOAK: no samples"); return 1
    first, last = samples[0], samples[-1]
    mid = samples[len(samples) // 2]

    def cam_series(cid, key):
        return [sm["cameras"].get(cid, {}).get(key) for sm in samples if cid in sm["cameras"]]

    rss = [sm["rss_mb"] for sm in samples if sm["rss_mb"] is not None]
    cam_ids = sorted(last["cameras"].keys())
    leaks = sum(sm["secret_leak"] or 0 for sm in samples)
    reconnect_max = {cid: max([x or 0 for x in cam_series(cid, "reconnects")] or [0]) for cid in cam_ids}
    tickerr_max = {cid: max([x or 0 for x in cam_series(cid, "tick_errors")] or [0]) for cid in cam_ids}
    model_loads = {sm["model_load_count"] for sm in samples if sm["model_load_count"] is not None}
    alive_always = all(all((sm["cameras"].get(cid, {}).get("alive")) for cid in cam_ids) for sm in samples)

    summary = {
        "samples": len(samples),
        "duration_s": round(last["t"] - first["t"], 1),
        "rss_begin_mb": first["rss_mb"], "rss_mid_mb": mid["rss_mb"], "rss_end_mb": last["rss_mb"],
        "rss_delta_mb": round((last["rss_mb"] - first["rss_mb"]), 1) if rss else None,
        "rss_peak_mb": max(rss) if rss else None,
        "queue_peak": max([sm["queue_peak"] or 0 for sm in samples]),
        "model_load_counts_seen": sorted(model_loads),
        "reconnect_max_per_cam": reconnect_max,
        "tick_errors_max_per_cam": tickerr_max,
        "workers_alive_throughout": alive_always,
        "secret_leaks_total": leaks,
        "overall_states_seen": sorted({sm["overall"] for sm in samples if sm["overall"]}),
        "fps_end_per_cam": {cid: last["cameras"].get(cid, {}).get("fps") for cid in cam_ids},
        "confirmed_falls_end": {cid: last["cameras"].get(cid, {}).get("confirmed_falls") for cid in cam_ids},
    }
    with open("diagnostics/soak_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SOAK SUMMARY:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
