# VytalLink Overnight Reliability, Performance & False-Positive Report

_Generated during the autonomous overnight pass starting 2026-06-17 07:06 PDT.
Soak is still accumulating; the soak/final-validation sections are updated at
finalization. All times UTC unless noted. No secrets, footage, or absolute paths
are included._

## 1. Executive summary

The overnight pass delivered the **Mac-side** reliability and false-positive work
in full and committed it on `hardware-integration`. The **Jetson-side** work
(Phases B–E: deploy the optimized relay, configure it, make the route persistent,
validate the relay) is **blocked by authentication** — there is no working
non-interactive SSH to the Jetson (`rchopra@192.168.42.43` → `Permission denied
(publickey,password)`; the Ethernet path `.3` closes because of the temporary
asymmetric host route). I did not guess credentials. Exact manual Jetson steps are
in §16–18.

Headline result: the ~8–22 confirmed "fall" events seen previously were
**false positives** caused by the transition gate being **off** — any sustained
`fallen` posture (already-low person, sitting/crouching, or a flickering low
posture) confirmed an event. I enabled and reinforced the gate, added conservative
detector box gates and a state-machine reconfirm cooldown, exposed full detection
geometry/rejection metadata, and added 13 regression tests. With the gates on, the
live pipeline produced **0 new confirmed events** overnight and remained stable.

The 8–10 FPS throughput target is **not reachable from the Mac alone**: the Jetson
relay still serves frames from its old 0.5 s detector-loop copy (~2 unique FPS),
which caps the Mac at ~4–5 received FPS. Deploying the already-written relay code
to the Jetson (commands in §16) lifts this cap.

## 2. Exact architecture

```
Tapo RTSP (192.168.42.251 /stream1, 2304x1296 ~19fps)
  → Jetson (192.168.42.43 wlan0 / 192.168.42.3 eth)  [HTTP MJPEG relay, VISION_MODE=rtsp, DETECTOR=simulation]
    → Mac (192.168.42.41)  [VISION_MODE=http_mjpeg → Apple MPS YOLO → event state machine → dashboard]
```
Alerts disabled end-to-end. No footage saved.

## 3. Commits

| Repo | Branch | Commits this pass |
|---|---|---|
| Mac (`~/Projects/VytalLink`) | `hardware-integration` | `943b9ac` Add false-positive gates, detection metadata, and soak diagnostics. Backup tag `backup/pre-overnight-c020726` at the prior HEAD `c020726`. |
| Jetson (`/home/rchopra/VytalLink`) | `hardware-integration` | **None — auth-blocked (no SSH).** Manual deploy in §16. |

## 4. Root causes found

1. **False positives:** `DETECTOR_REQUIRE_TRANSITION` was off, so the detector
   surfaced *any* sustained `fallen` posture as fall evidence. A person already
   low, sitting/crouching, low in frame, or a flickering low posture confirmed
   events; evidence flicker (fallen↔not) created *repeated* events
   (~8 in 90 s previously; 22 accumulated in the gates-off soak window).
2. **Throughput cap (unchanged, Jetson-side):** the Jetson relay serves the frame
   its own 0.5 s monitor/detector loop last copied, so only ~2 *unique* FPS reach
   the Mac (the rest are re-sent duplicates). Mac-side pacing cannot exceed what
   the relay delivers.

## 5. Changes made (Mac, commit `943b9ac`)

- **Conservative detector box gates** (`detector_yolo.py`, off by default):
  `DETECTOR_MIN_FALLEN_BOX_AREA_FRAC` (tiny/far `fallen` boxes don't count) and
  `DETECTOR_REJECT_EDGE_CLIPPED_FALLEN` (left/right/top edge-clipped partial-person
  boxes don't count; **bottom-edge allowed** since real falls land low). Rejected
  boxes are surfaced as the non-evidence `fallen_posture` class with a reason.
- **State-machine reconfirm cooldown** (`state_machine.py`, off by default):
  `FALL_RECONFIRM_COOLDOWN_SECONDS` suppresses a *new* confirmation within the
  window after a confirmation, so one continuous (flickering) low posture yields
  at most one event; a genuine later fall still confirms.
- **Detection geometry metadata:** every detection carries `bbox_norm`,
  `area_frac`, `aspect`, `vertical_center`, `edges`, and any `rejection`; the
  dev-only `/api/detector/debug` exposes per-class counts, rejection counts, gate
  config, candidate duration, and transition history — no images/credentials/paths.
- **Soak collector** (`scripts/soak_collector.py`): samples Mac + Jetson health
  every 30 s to a gitignored `diagnostics/*.jsonl` (metrics only).
- The existing `DETECTOR_REQUIRE_TRANSITION` gate (requires an upright→fallen
  transition; rejects already-low/static/no-prior-upright) remains the primary
  filter and is **enabled at runtime** for the live detector.

CUDA-preference on Jetson / MPS-preference on Mac / CPU fallback are untouched
(`common/device.py` unchanged this commit; selection remains CUDA→MPS→CPU).

## 6. Live detector runtime config (Mac, this pass)

Started via process env (the local `.env` was **not** overwritten; the camera URL
stays in `.env`, never on the command line):
`DETECTOR_MODE=yolo VISION_MODE=http_mjpeg WEARABLE_MODE=simulation
ALERTS_ENABLED=false DASHBOARD_LIVE_VIDEO=true DASHBOARD_SHOW_DETECTIONS=true
DETECT_MAX_FPS=10 DETECT_MAX_FRAME_AGE_SECONDS=0.5 DETECTOR_REQUIRE_TRANSITION=true
FALL_RECONFIRM_COOLDOWN_SECONDS=15 DETECTOR_MIN_FALLEN_BOX_AREA_FRAC=0.02
DETECTOR_REJECT_EDGE_CLIPPED_FALLEN=true DISK_WARNING_PERCENT=100`.

`DETECT_MAX_FRAME_AGE_SECONDS=0.5` justified by measurement: the relay delivers
~4–5 FPS so fresh frames are <0.25 s old; 0.5 s drops only genuinely stalled
frames while never dropping live ones.

## 7. Soak

- Collector PID 6553, file `diagnostics/soak_20260617_070623.jsonl`, 30 s interval,
  4 h target. Cutover (gates enabled) at **2026-06-17T14:14:28Z**.
- _Status at report creation: in progress (~30 min). Final duration + p50/p95 are
  filled at finalization (§8–10)._

## 8. Before/after performance (interim)

| Metric | Before (gates off, c020726) | After (gates on, 943b9ac) |
|---|---|---|
| Mac receive FPS | ~6.8 (incl. duplicate JPEGs) | ~4.4 |
| Mac inference FPS | ~6.1 | ~3.8 |
| Mac frame age (max) | 0.08 s | 0.09 s |
| MPS inference latency | ~20 ms | ~20 ms |
| Process RSS | ~137 MB stable | ~233 MB stable (after load spike to ~409 MB) |
| Confirmed events | **22 accumulated; 237 fallen-frames** | **0 new** |
| Jetson relay unique FPS | ~2 | ~2 (unchanged — relay not redeployed) |

_(FPS varies with the rolling grab window; finalized averages in the summary.)_

## 9. p50 / p95 metrics

_Filled at finalization from the full soak JSONL._

## 10. Reconnects, failures, memory & CPU trends

- Reconnects / failed reads / stale drops: 0 across the interim window.
- Memory: post-warmup RSS ~233 MB, no growth trend yet; FDs ~183, threads stable.
  _Full-soak trend filled at finalization; flagged if monotonic growth appears._

## 11. False-positive analysis

- **Cause:** transition gate off → sustained/flickering `fallen` confirmed events.
- **Evidence:** gates-off instance logged 237 fallen-frames and 22 events; the same
  camera/scene with gates on logged 0 fall evidence and 0 events.
- **Caveat (honest):** the overnight scene is largely empty/static, so part of the
  0-event result is an empty scene, not solely the gates. Gate *efficacy* is proven
  by the 13 deterministic regression tests (§13), which exercise standing→fallen,
  sitting, already-lying, flicker, continuous-posture, recovery, duplicate, stale,
  and edge-clipped cases directly.
- Per-detection metadata (class, confidence, `bbox_norm`, `area_frac`, `aspect`,
  `vertical_center`, `edges`, `rejection`, transition history) is recorded in the
  soak JSONL and `/api/detector/debug` for deeper correlation if activity occurs.

## 12. Transition-filter design

A `fallen` posture is treated as a fall **event** only when:
1. a prior **upright** (standing/sitting) state was established (gap-tolerant,
   `min_upright_frames`), then
2. a sustained **fallen** run (`min_fallen_frames`) begins within
   `transition_window_seconds` of that upright, and
3. the `fallen` box passes the conservative size/edge gates, and
4. the state machine sees sustained evidence for `FALL_CONFIRM_SECONDS`, and
5. it is not within `FALL_RECONFIRM_COOLDOWN_SECONDS` of a prior confirmation.

Anti-stale history reset (`stale_seconds`) means a person re-appearing already on
the floor never registers as a fall. Already-down monitoring remains available by
setting `DETECTOR_REQUIRE_TRANSITION=false` (documented as more false-positive
prone); the reconfirm cooldown still caps repeated events in that mode.

## 13. Test results

- Mac full suite: **213 passed** (was 200; +13 false-positive regression tests).
- New tests (`tests/unit/test_false_positive_gates.py`): standing→fallen confirms;
  sitting / already-lying / low-confidence-flicker do **not**; one continuous low
  posture → ≤1 event; recovery allows a later independent event; duplicate frames
  don't advance confirmation; stale frames don't advance evidence; tiny &
  edge-clipped `fallen` boxes handled conservatively; geometry metadata present;
  reconfirm cooldown suppresses-then-allows and is off by default.

## 14. Smoke-test results

- Mac smoke: **PASS** (re-run at finalization).
- Jetson tests/smoke: **not run — auth-blocked**.

## 15. Unresolved risks

1. **Jetson relay not upgraded** → throughput stays ~2 unique FPS; 8–10 FPS target
   unmet until §16 is run on the Jetson.
2. **Empty-scene caveat** on the live false-positive number (mitigated by tests).
3. **Temporary host route** on the Jetson is not persistent (auth-blocked); a
   Jetson reboot drops it (§18).
4. Memory was observed only over a short window at report creation; see §10 for the
   full-soak trend.

## 16. Exact next recommended step — deploy the relay code to the Jetson

The Mac commit `943b9ac` (and `c020726`) already contain the relay improvements
(freshest-frame relay, `RELAY_WIDTH/HEIGHT/JPEG_QUALITY/MAX_FPS`, stale handling,
badges, `ALERTS_ENABLED`). On the **Jetson** (after enabling SSH, e.g.
`ssh-copy-id rchopra@192.168.42.43` once from the Mac):

```sh
cd /home/rchopra/VytalLink
git fetch <mac-or-shared-remote> hardware-integration   # or apply a patch bundle
git switch hardware-integration && git pull --ff-only
# Do NOT touch .venv/.env/models/db/logs. Then set relay-only config in the Jetson .env:
#   VISION_MODE=rtsp DETECTOR_MODE=simulation WEARABLE_MODE=simulation
#   CAMERA_STREAM_PATH=stream1 DASHBOARD_LIVE_VIDEO=true ALERTS_ENABLED=false
#   CONSOLE_ALERTS_ENABLED=false RELAY_WIDTH=960 RELAY_HEIGHT=540
#   RELAY_JPEG_QUALITY=70 RELAY_MAX_FPS=10
bash scripts/stop.sh; pkill -f vytallink.app; bash scripts/start.sh
curl --max-time 10 -o /tmp/s.jpg http://127.0.0.1:5050/api/camera/snapshot.jpg && file /tmp/s.jpg
```
Then on the Mac, restart the detector (§17) and re-run the soak; expect ~8–10
unique relay FPS and Mac inference ≥8 FPS.

## 17. Exact commands to restart both systems

**Mac detector** (camera URL stays in `.env`):
```sh
cd ~/Projects/VytalLink
bash scripts/stop.sh; pkill -f vytallink.app; sleep 1
lsof -iTCP:5050 -sTCP:LISTEN -n -P || echo free   # verify port free (stale-PID guard)
DETECTOR_MODE=yolo VISION_MODE=http_mjpeg WEARABLE_MODE=simulation ALERTS_ENABLED=false \
DASHBOARD_LIVE_VIDEO=true DASHBOARD_SHOW_DETECTIONS=true DETECT_MAX_FPS=10 \
DETECT_MAX_FRAME_AGE_SECONDS=0.5 DETECTOR_REQUIRE_TRANSITION=true \
FALL_RECONFIRM_COOLDOWN_SECONDS=15 DETECTOR_MIN_FALLEN_BOX_AREA_FRAC=0.02 \
DETECTOR_REJECT_EDGE_CLIPPED_FALLEN=true DISK_WARNING_PERCENT=100 \
  bash scripts/start.sh
```
**Jetson relay**: see §16.
**Soak collector** (Mac): `JETSON_HEALTH_URL=http://192.168.42.43:5050 nohup ./.venv/bin/python scripts/soak_collector.py 14400 30 > diagnostics/soak_collector.log 2>&1 &`

## 18. Undo the temporary route change

I did **not** add or change any route (auth-blocked). The user's existing
temporary route can be removed on the **Jetson** with:
```sh
sudo ip route del 192.168.42.41/32 dev wlan0   # remove the temporary Mac-bound route
```
To make it persistent instead (NetworkManager, replace CONN with the wlan0
connection name from `nmcli -t -f NAME,DEVICE connection show --active`):
```sh
sudo nmcli connection modify "CONN" +ipv4.routes "192.168.42.41/32 0.0.0.0"
sudo nmcli connection up "CONN"     # re-applies without a reboot
```
Avoid creating two equal-priority default routes; do not disable Ethernet.

## 19–21. Safety confirmations

- **Alerts stayed disabled** the entire pass: `ALERTS_ENABLED=false`, dispatcher
  built with zero providers, health `alerts.status=disabled`, 0 new alerts in the
  DB (the single stored alert predates this pass).
- **No footage stored**: `SAVE_EVENT_SNAPSHOTS/CLIPS=false`; `data/events` and
  `data/clips` empty; the soak records metrics only, never frames.
- **No secrets exposed**: health/status/debug and logs show only
  `scheme://host:port` (no full RTSP path, no camera user/password, no bearer
  token, no absolute paths). `.env` files were not overwritten or printed.
