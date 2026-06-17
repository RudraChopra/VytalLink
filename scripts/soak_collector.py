#!/usr/bin/env python3
"""VytalLink overnight soak metrics collector (metrics only — never images).

Samples the Mac detector app (127.0.0.1:5050) and, best-effort, the Jetson relay
(JETSON_HEALTH_URL) every INTERVAL seconds, appending one compact JSON object per
sample to a JSONL file under diagnostics/. Records lightweight reliability /
performance / false-positive metrics. Never stores raw frames. Never prints
secrets (only host:port from already-sanitized health is read).

Usage:
  soak_collector.py [duration_seconds] [interval_seconds]
Env:
  MAC_HEALTH_URL     (default http://127.0.0.1:5050)
  JETSON_HEALTH_URL  (default http://192.168.42.43:5050; set empty to skip)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAC = os.environ.get("MAC_HEALTH_URL", "http://127.0.0.1:5050")
JET = os.environ.get("JETSON_HEALTH_URL", "http://192.168.42.43:5050")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)


def _get(base: str, path: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.load(r)
    except Exception as exc:  # never crash the collector on a transient error
        return {"_error": type(exc).__name__}


def _pid_on_5050() -> int | None:
    try:
        out = subprocess.run(
            ["lsof", "-iTCP:5050", "-sTCP:LISTEN", "-n", "-P", "-Fp"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            if line.startswith("p"):
                return int(line[1:])
    except Exception:
        pass
    return None


def _proc_stats(pid: int | None) -> dict:
    if not pid:
        return {"rss_mb": None, "cpu_pct": None, "num_fds": None, "num_threads": None}
    rss_mb = cpu = None
    try:
        out = subprocess.run(["ps", "-o", "rss=,%cpu=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.split()
        if len(out) >= 2:
            rss_mb = round(int(out[0]) / 1024, 1)
            cpu = float(out[1])
    except Exception:
        pass
    nfds = nthreads = None
    try:
        nfds = len(subprocess.run(["lsof", "-p", str(pid)], capture_output=True,
                                  text=True, timeout=8).stdout.splitlines()) - 1
    except Exception:
        pass
    try:
        nthreads = int(subprocess.run(["ps", "-o", "thcount=", "-p", str(pid)],
                                      capture_output=True, text=True, timeout=5).stdout.strip() or 0)
    except Exception:
        pass
    return {"rss_mb": rss_mb, "cpu_pct": cpu, "num_fds": nfds, "num_threads": nthreads}


def sample() -> dict:
    now = datetime.now(timezone.utc)
    mac = _get(MAC, "/health")
    dbg = _get(MAC, "/api/detector/debug")
    jet = _get(JET, "/health") if JET else {"_skipped": True}
    pid = _pid_on_5050()
    proc = _proc_stats(pid)
    du = shutil.disk_usage(str(ROOT))
    mc = mac.get("camera", {}) if isinstance(mac, dict) else {}
    md = mac.get("detector", {}) if isinstance(mac, dict) else {}
    jc = jet.get("camera", {}) if isinstance(jet, dict) else {}
    return {
        "t": now.isoformat(),
        "mac_pid": pid,
        # Mac pipeline
        "mac_mode": mac.get("mode") if isinstance(mac, dict) else None,
        "mac_overall": mac.get("overall") if isinstance(mac, dict) else None,
        "mac_recv_fps": mc.get("effective_fps"),
        "mac_frame_age": mc.get("last_frame_age_seconds"),
        "mac_dropped": mc.get("frames_dropped"),
        "mac_dropped_stale": mc.get("frames_dropped_stale"),
        "mac_failed_reads": mc.get("failed_reads"),
        "mac_reconnects": mc.get("reconnects"),
        "mac_frames_processed": mc.get("frames_processed"),
        "inf_fps": md.get("inference_fps"),
        "inf_avg_ms": md.get("avg_inference_ms"),
        "inf_last_ms": md.get("last_inference_ms"),
        "inf_count": md.get("inference_count"),
        "device": md.get("device"),
        "fall_state": mac.get("fall_state") if isinstance(mac, dict) else None,
        "alerts": (mac.get("alerts", {}) or {}).get("status") if isinstance(mac, dict) else None,
        # detector debug (false-positive analysis)
        "class_counts": dbg.get("class_counts") if isinstance(dbg, dict) else None,
        "last_detections": dbg.get("last_detections") if isinstance(dbg, dict) else None,
        "frames_with_fallen": dbg.get("frames_with_fallen") if isinstance(dbg, dict) else None,
        "evidence_score": dbg.get("evidence_score") if isinstance(dbg, dict) else None,
        "fall_candidate_seconds": dbg.get("fall_candidate_seconds") if isinstance(dbg, dict) else None,
        "rejections": dbg.get("rejections") if isinstance(dbg, dict) else None,
        # Jetson relay (remote, best-effort)
        "jet_overall": jet.get("overall") if isinstance(jet, dict) else None,
        "jet_cam_status": jc.get("status"),
        "jet_recv_fps": jc.get("effective_fps"),
        "jet_unique_fps": jc.get("unique_fps"),
        "jet_frame_age": jc.get("last_frame_age_seconds"),
        "jet_frames": jc.get("frame_count"),
        "jet_reconnects": jc.get("reconnects"),
        "jet_resolution": jc.get("resolution"),
        # host
        "proc": proc,
        "disk_pct": round(100 * (du.total - du.free) / du.total, 1),
    }


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 14400.0
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"soak_{stamp}.jsonl"
    meta = {"_meta": True, "started": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration, "interval_s": interval, "mac": MAC, "jetson": JET}
    with out.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        f.flush()
        end = time.monotonic() + duration
        n = 0
        while time.monotonic() < end:
            t0 = time.monotonic()
            try:
                f.write(json.dumps(sample()) + "\n")
                f.flush()
                n += 1
            except Exception as exc:
                f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(),
                                    "_sample_error": type(exc).__name__}) + "\n")
                f.flush()
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
    print(f"soak done: {n} samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
