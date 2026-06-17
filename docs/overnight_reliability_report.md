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

Two headline results:
1. **Reliability:** found and fixed a reproducible **MPS/Metal thread-safety crash**
   — inference on `asyncio.to_thread`'s multi-worker pool aborted the process
   intermittently. Pinning all accelerator work to one dedicated thread eliminated
   it (verified live: pre-fix run aborted, post-fix run stable). The 4 h soak +
   15 min post-fix validation showed **no crash, no memory leak, no FD leak**.
2. **False positives:** the ~8–22 prior "fall" events were caused by the
   transition gate being **off** — any sustained `fallen` posture (already-low,
   sitting/crouching, or flickering) confirmed an event. I enabled/reinforced the
   gate, added conservative detector box gates and a reconfirm cooldown, exposed
   detection geometry/rejection metadata, and added regression tests. With gates
   on: **0 false events** across the 4 h soak and 15 min validation.

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
| Mac (`~/Projects/VytalLink`) | `hardware-integration` | `943b9ac` false-positive gates + detection metadata + soak diagnostics; `f557ce4` report; `7465ea9` MPS single-thread crash fix + grab-based frame age + bounded relay backoff. Backup tag `backup/pre-overnight-c020726` at the prior HEAD `c020726`. |
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
3. **MPS thread-safety crash (reproducible, now fixed):** inference ran via
   `asyncio.to_thread`, whose multi-worker pool let consecutive MPS inferences
   land on different threads. Metal command buffers are not thread-safe across
   threads, so the process intermittently aborted with
   `-[_MTLCommandBuffer addScheduledHandler:]:807: failed assertion 'Scheduled
   handler provided after commit call'`. The 4 h soak narrowly avoided it; an app
   restart reproduced it. Fixed by pinning all accelerator work to one dedicated
   thread (commit `7465ea9`); verified live (pre-fix run aborted, post-fix run
   stable).
4. **Misleading frame-age metric (fixed):** `last_frame_age_seconds` was
   consumer-read-based, so it inflated to 378 s p95 / 1068 s max during reconnect
   flaps even while frames flowed. Now grab-based for buffered cameras.
5. **Reconnect flaps (network):** reconnects grew 2→113 over 4 h, concentrated in
   ~8 % of samples where the Jetson `/health` (small) succeeded but the bulk MJPEG
   stream stalled — the known Wi-Fi bulk-transfer limitation. Self-healing
   (92 % of samples fresh); relay backoff lowered to 5 s for faster recovery.

## 5. Changes made (Mac)

**Reliability (commit `7465ea9`):**
- **MPS single-thread pinning** — one dedicated `ThreadPoolExecutor(max_workers=1)`
  owns model load, warmup, and every inference, so Metal command buffers never
  cross threads (fixes the abort in §4.3).
- **Grab-based frame age** for buffered cameras (HTTP relay, RTSP) so freshness is
  accurate during reconnects (§4.4).
- **5 s reconnect-backoff cap** on the HTTP relay camera (LAN reconnect is cheap)
  for fast recovery from a flapping link (§4.5).

**False-positive reduction (commit `943b9ac`)**

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

- Collector PID 6553, `diagnostics/soak_20260617_070623.jsonl`, 30 s interval.
  **Completed: 3.99 h, 480 samples, 0 sample errors, 0 app restarts.** Cutover
  (gates enabled) at **2026-06-17T14:14:28Z** (17 before, 463 after).
**Final 15-min validation (post-fix `7465ea9`, 59 samples):** stable, no crash
(17-min continuous uptime, 0 Metal assertions after the fix), no reconnects, no
failed reads, RSS settled ~135 MB, FDs flat (182). Mac receive FPS p50 **6.56**
/ p95 6.97; Mac inference FPS p50 **6.37** / p95 6.82; MPS latency p50 **23.8 ms**
/ p95 28.8 ms; **grab-based frame age p50 0.06 s / p95 0.18 s / max 0.23 s**
(metric now correct); fall_state `normal` for all 59 samples (**0 false events**);
alerts disabled; Jetson relay ~**2** unique FPS (still the cap).

## 8. Before/after performance (4 h soak)

| Metric | Before (gates off, `c020726`) | After (gates on, `943b9ac`) |
|---|---|---|
| Mac receive FPS | ~6.8 (incl. duplicate JPEGs) | p50 6.7 / p95 7.26 |
| Mac inference FPS | ~6.1 | p50 6.31 / p95 6.85 |
| MPS inference latency | ~20 ms | p50 18.2 / p95 23.1 ms |
| Process RSS | ~137 MB stable | 409 MB at load → settled ~135 MB (no leak) |
| Confirmed false events | **22 accumulated; 237 fallen-frames** | **0 over 3.99 h** |
| Box-gate rejections | n/a (gates off) | `too_small`:6, `edge_clipped_t`:6 (working on real activity) |
| Jetson relay unique FPS | ~2 | ~2 (unchanged — relay not redeployed) |

## 9. p50 / p95 metrics (4 h soak, after-cutover)

- Mac receive FPS: p50 **6.7**, p95 **7.26**, max 8.34.
- Mac inference FPS: p50 **6.31**, p95 **6.85**.
- MPS inference latency: p50 **18.2 ms**, p95 **23.1 ms** (one 669 ms outlier).
- Frame age: p50 **0.06 s**; p95/max were inflated (378 s / 1068 s) by the
  read-based metric during reconnect flaps — **fixed** to grab-based in `7465ea9`;
  the §J validation reports the corrected metric.

## 10. Reconnects, failures, memory & CPU trends

- **Memory: NO leak** — RSS 409 MB at model load, settled to ~135 MB (min 57),
  last 139 MB; net negative over 4 h. **FDs: 181→182 (no leak).** No thread leak.
- Failed reads: **0**. Stale drops: 2. Disk: 88.9 %.
- **Reconnects: 2→113**, concentrated in ~8 % of samples (stall windows). The
  Jetson `/health` reported `ok` throughout, so the small control plane was fine
  while the bulk MJPEG stream flapped — the known Wi-Fi bulk-transfer issue. The
  pipeline self-healed (92 % of samples had fresh <1 s frames; inference p50
  6.3 FPS). Mitigations: grab-based age (accurate freshness) + 5 s backoff cap.

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

- Mac full suite: **216 passed** (was 200; +13 false-positive + 3 reliability
  regression tests: MPS single-thread pinning, grab-based frame age, bounded
  relay backoff).
- New tests (`tests/unit/test_false_positive_gates.py`): standing→fallen confirms;
  sitting / already-lying / low-confidence-flicker do **not**; one continuous low
  posture → ≤1 event; recovery allows a later independent event; duplicate frames
  don't advance confirmation; stale frames don't advance evidence; tiny &
  edge-clipped `fallen` boxes handled conservatively; geometry metadata present;
  reconfirm cooldown suppresses-then-allows and is off by default.

## 14. Smoke-test results

- Mac smoke: **PASS** (re-run after every code change, incl. the MPS fix).
- Jetson tests/smoke: **not run — auth-blocked** (no SSH).

## 15. Targets (Phase J §72) and unresolved risks

Targets met / not met:
- ✅ p95 source→inference frame age < 500 ms → **0.18 s** (max 0.23 s).
- ✅ no growing queue; ✅ no unbounded memory (RSS net-negative over 4 h); ✅ no
  repeated event from one continuous low posture; ✅ alerts disabled; ✅ no footage;
  ✅ no secrets leaked.
- ❌ Jetson relay ≥8 / Mac receive ≥8 / Mac inference ≥8 unique FPS — **limited by
  the Jetson relay (~2 unique FPS), the proven limiting stage.** Mac receive ~6.6
  and inference ~6.4 include duplicate JPEGs the relay re-sends; *unique* content
  is ~2 FPS. Lifted only by deploying the relay code to the Jetson (§16).

Unresolved risks:
1. **Jetson relay not upgraded** (auth blocker) → ~2 unique FPS until §16 is run.
2. **Wi-Fi bulk-transfer flaps** (reconnect storms) recur intermittently; resolved
   by the Jetson Ethernet link + new relay. The Mac self-heals (5 s backoff).
3. **Empty-scene caveat** on the live false-positive number (mitigated by the 13
   deterministic regression tests; box gates also fired on real activity).
4. **Temporary host route** on the Jetson is not persistent (auth-blocked); a
   Jetson reboot drops it (§18).

## 16. Exact next recommended step — deploy the relay code to the Jetson

The Mac commit `943b9ac` (and `c020726`) already contain the relay improvements
(freshest-frame relay, `RELAY_WIDTH/HEIGHT/JPEG_QUALITY/MAX_FPS`, stale handling,
badges, `ALERTS_ENABLED`). On the **Jetson** (after enabling SSH, e.g.
`ssh-copy-id rchopra@192.168.42.43` once from the Mac):

A transferable git bundle was prepared on the Mac (no remote needed):
`diagnostics/vytallink-hardware-integration.bundle` (contains `c020726` +
`943b9ac`). Copy it to the Jetson (scp/USB) and fast-forward:
```sh
cd /home/rchopra/VytalLink
# scp it over first, e.g. to /tmp/vl.bundle, then:
git bundle verify /tmp/vl.bundle
git fetch /tmp/vl.bundle hardware-integration:refs/remotes/macbundle/hi
git switch hardware-integration && git merge --ff-only macbundle/hi   # same branch, fast-forward only
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
