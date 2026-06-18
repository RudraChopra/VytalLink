# VytalLink Overnight Reliability, Performance & False-Positive Report

_Generated during the autonomous overnight pass starting 2026-06-17 07:06 PDT;
finalized 2026-06-17 ~18:15 PDT after the Jetson deployment completed. All times
UTC unless noted. No secrets, footage, or absolute paths are included._

## 1. Executive summary

The overnight pass is **complete on both machines**. The Mac-side reliability and
false-positive work landed and was committed on `hardware-integration`; the
optimized relay was then deployed to the **Jetson** (once SSH was enabled),
configured relay-only, and validated live end-to-end.

Headline results:
1. **Reliability:** found and fixed a reproducible **MPS/Metal thread-safety crash**
   — inference on `asyncio.to_thread`'s multi-worker pool aborted the process
   intermittently. Pinning all accelerator work to one dedicated thread eliminated
   it (verified live: pre-fix run aborted, post-fix run stable). The 4 h soak +
   the final 15 min live E2E showed **no crash, no memory leak, no FD leak,
   0 reconnects**.
2. **False positives:** the ~8–22 prior "fall" events were caused by the
   transition gate being **off** — any sustained `fallen` posture (already-low,
   sitting/crouching, or flickering) confirmed an event. I enabled/reinforced the
   gate, added conservative detector box gates and a reconfirm cooldown, exposed
   detection geometry/rejection metadata, and added regression tests. With gates
   on: **0 false events** across the 4 h soak and the final live E2E.
3. **Throughput target met (resolved):** the old Jetson relay served frames from
   its 0.5 s detector-loop copy (~2 unique FPS), capping the Mac. Deploying the
   freshest-frame relay to the Jetson and configuring it relay-only lifted the cap
   to **~8.3 unique relay FPS → 8.2 Mac receive FPS → 7.2 Mac MPS inference FPS**,
   with p95 source→inference frame age **0.19 s** and p95 MPS latency **33.5 ms**.

**One environmental surprise, fixed:** the Tapo camera had moved by DHCP from
`192.168.42.251` to **`192.168.42.71`** (the old address was dead — ARP
incomplete from both the Mac and the Jetson). A bounded RTSP sweep located the new
address (confirmed `RTSP/1.0 200 OK`); the Jetson `.env` `CAMERA_HOST` was rewritten
in place (embedded credentials preserved, `.env` backed up first). The new address
is reachable over the Jetson's default route, so **no route change or `sudo` was
required** (see §4/§18).

## 2. Exact architecture

```
Tapo RTSP (192.168.42.71 /stream1, 2304x1296, grabber ingest ~15fps)   [was .251; moved by DHCP]
  → Jetson (192.168.42.43 wlan0 / 192.168.42.3 eth0)  [HTTP MJPEG relay, VISION_MODE=rtsp, DETECTOR=simulation,
                                                       RELAY 960x540 q70 @ ≤10fps → ~8.3 unique fps out]
    → Mac (192.168.42.41)  [VISION_MODE=http_mjpeg → Apple MPS YOLO → event state machine → dashboard]
```
Alerts disabled end-to-end. No footage saved. The Jetson is dual-homed on the same
/24 (eth0 `.3`, wlan0 `.43`); the camera at `.71` is reachable via the default
(eth0) route — the wired and Wi-Fi segments are bridged — so no host route is
needed for it. A pre-existing temporary `/32` route pins Jetson→Mac traffic to
wlan0 (not required for the relay; the Mac is the client).

## 3. Commits

| Repo | Branch | Commits this pass |
|---|---|---|
| Mac (`~/Projects/VytalLink`) | `hardware-integration` | `943b9ac` false-positive gates + detection metadata + soak diagnostics; `f557ce4` report; `7465ea9` MPS single-thread crash fix + grab-based frame age + bounded relay backoff; `50c9833` finalized report. Backup tag `backup/pre-overnight-c020726` at the prior HEAD `c020726`. **HEAD = `50c9833`.** Not pushed/merged. |
| Jetson (`/home/rchopra/VytalLink`) | `hardware-integration` | Fast-forwarded `c2cc771 → 50c9833` via an offline git **bundle** (`git fetch <bundle>` + `git merge --ff-only`; no remote, no push). `.env`/`.venv`/models/db/logs untouched (gitignored). Backup tag `backup/pre-overnight-c2cc771` at the prior HEAD `c2cc771`. **HEAD = `50c9833`** (identical to Mac). |

## 4. Root causes found

1. **False positives:** `DETECTOR_REQUIRE_TRANSITION` was off, so the detector
   surfaced *any* sustained `fallen` posture as fall evidence. A person already
   low, sitting/crouching, low in frame, or a flickering low posture confirmed
   events; evidence flicker (fallen↔not) created *repeated* events
   (~8 in 90 s previously; 22 accumulated in the gates-off soak window).
2. **Throughput cap (Jetson-side, now RESOLVED):** the *old* Jetson relay served
   the frame its own 0.5 s monitor/detector loop last copied, so only ~2 *unique*
   FPS reached the Mac (the rest were re-sent duplicates). Deploying the
   freshest-frame relay (`peek_latest()`-driven, paced at `RELAY_MAX_FPS`) and
   running the Jetson relay-only lifted this to **~8.3 unique FPS** out of the
   relay. The grabber now ingests the camera at ~15 FPS and the relay serves a
   fresh paced subset (15 > 10, so every served frame is new).
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
   *During the Jetson cutover the same flap made interactive SSH drop mid-session;
   mitigated by driving the deploy/fix/validate as a single detached remote script
   polled for results, and by stopping the (starved) Mac client to shed load.*
6. **Camera moved by DHCP (`.251` → `.71`), and Jetson dual-homing:** the Tapo was
   no longer at `192.168.42.251` (ARP incomplete from both hosts on the Wi-Fi LAN).
   A bounded TCP/554 sweep + an RTSP `OPTIONS` handshake located it at
   `192.168.42.71` (`RTSP/1.0 200 OK`). Fixed by rewriting `CAMERA_HOST` in the
   Jetson `.env` (credentials preserved; `.env` backed up). The Jetson is dual-homed
   on the same /24 (eth0 `.3` preferred, metric 100; wlan0 `.43`, metric 600); the
   new camera address resolves over the default eth0 route (segments bridged), so
   **no `/32` route or `sudo` was needed**. Had it not, a wlan0 `/32` route would
   have been required — and `sudo` needs a password on the Jetson, which would have
   been a user-action blocker.

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
- Jetson full suite (post-deploy, same `50c9833`): **216 passed** (13.6 s) on its
  Python 3.10 `--system-site-packages` venv — identical count, new modules import
  cleanly on the Jetson.
- New tests (`tests/unit/test_false_positive_gates.py`): standing→fallen confirms;
  sitting / already-lying / low-confidence-flicker do **not**; one continuous low
  posture → ≤1 event; recovery allows a later independent event; duplicate frames
  don't advance confirmation; stale frames don't advance evidence; tiny &
  edge-clipped `fallen` boxes handled conservatively; geometry metadata present;
  reconfirm cooldown suppresses-then-allows and is off by default.

## 14. Smoke-test results

- Mac smoke: **PASS** (all 22 checks; re-run after every code change incl. the MPS
  fix, and once more post-Jetson-deploy on port 5077 without disturbing the live
  pipeline).
- Jetson smoke: **PASS** (all 22 checks; ran on port 5077 in simulation mode, so
  the live relay on 5050 was untouched).

## 15. Final live E2E results (15 min, after Jetson deploy) and targets

Final 15-minute end-to-end soak with the upgraded Jetson relay (45 samples @ 20 s,
`diagnostics/soak_20260617_180026.jsonl`, summary in
`diagnostics/final_e2e_analysis.txt`):

| Metric | p50 | p95 | max | notes |
|---|---|---|---|---|
| Jetson grabber ingest FPS | 15.0 | 15.06 | 15.12 | full camera read |
| **Unique relay output FPS** | **~8.3** | — | — | direct MJPEG boundary count; `RELAY_MAX_FPS=10` |
| **Mac receive FPS** (unique) | **8.2** | 8.47 | 8.58 | distinct frames the Mac decodes |
| **Mac MPS inference FPS** | **7.19** | 7.76 | 7.96 | bounded by ~8.2 unique input |
| MPS inference latency (ms) | 25.0 | **33.5** | 36.7 | per-frame |
| Mac frame age (s) | 0.08 | **0.19** | 0.23 | source→inference freshness |
| Jetson frame age (s) | 0.04 | 0.07 | 0.07 | grabber freshness |

Reliability over the window: **0 Mac reconnects, 0 Jetson reconnects, 0 stale
drops, 0 failed reads, Mac PID stable (no crash), 0/45 non-`ok` health samples on
either machine.** False positives: `fall_state` was `normal` in all 45 samples,
`frames_with_fallen` cumulative **0**. Alerts `disabled`; device `mps`.

Targets (Phase J):
- ✅ Jetson relay ≥ 8 unique FPS → **~8.3**.
- ✅ Mac receive ≥ 8 FPS → **8.2** (p50).
- 🟡 Mac inference ≥ 8 FPS → **7.2** (p50; p95 7.76). Bounded by the ~8.2 unique
  input minus the few frames dropped for `age > 0.5 s` during minor jitter — within
  ~1 FPS of target and limited by input, not by MPS (which runs in ~25 ms ≈ 40 FPS
  headroom). Raising `DETECT_MAX_FRAME_AGE_SECONDS` slightly or `RELAY_MAX_FPS` to
  12 would close the gap.
- ✅ p95 source→inference frame age < 500 ms → **0.19 s** (max 0.23 s).
- ✅ no growing queue; ✅ no unbounded memory; ✅ no repeated event from one
  continuous low posture; ✅ alerts disabled; ✅ no footage; ✅ no secrets leaked.

Unresolved risks:
1. **Wi-Fi bulk-transfer flaps** still recur intermittently (they made interactive
   SSH drop during the cutover). The relay/Mac self-heal (5 s backoff); the final
   15 min window saw 0 reconnects. A wired-only path for the Mac would remove it.
2. **Camera on DHCP** — it already moved once (`.251`→`.71`). A DHCP reservation
   for the Tapo (or a hostname) would prevent a silent re-break; today it is pinned
   only by the `.env` literal IP.
3. **Empty-scene caveat** on the live false-positive number (mitigated by the 13
   deterministic regression tests; box gates also fired on real activity).
4. **Jetson→Mac `/32` route is live but not NetworkManager-persistent** (making it
   persistent needs `sudo`, which requires a password). It is **not** required for
   the relay (the Mac is the client; the camera path uses the default eth0 route),
   so this is cosmetic for relay operation — see §18.

## 16. Jetson deployment — COMPLETED

Performed this pass (no push, no merge, no remote):

1. **Backup tag** `backup/pre-overnight-c2cc771` at the prior Jetson HEAD.
2. **Code** fast-forwarded `c2cc771 → 50c9833` via an offline bundle
   (`diagnostics/vytallink-hardware-integration.bundle`, 76 K, regenerated to the
   Mac tip): `git fetch <bundle> hardware-integration` → `git merge --ff-only
   FETCH_HEAD`. `.env`/`.venv`/models/db/logs left untouched (gitignored).
3. **Relay-only `.env`** via targeted in-place edits (a fresh `.env.bak.overnight.*`
   was written first; camera credentials and `CAMERA_STREAM_PATH=stream1`
   preserved): `VISION_MODE=rtsp`, `DETECTOR_MODE=simulation`,
   `WEARABLE_MODE=simulation`, `ALERTS_ENABLED=false`,
   `CONSOLE_ALERTS_ENABLED=false`, `WEBHOOK_ALERTS_ENABLED=false`,
   `DASHBOARD_LIVE_VIDEO=true`, `RELAY_WIDTH=960`, `RELAY_HEIGHT=540`,
   `RELAY_JPEG_QUALITY=70`, `RELAY_MAX_FPS=10`, `SAVE_EVENT_SNAPSHOTS=false`,
   `SAVE_EVENT_CLIPS=false`. The Jetson runs **no YOLO** (`DETECTOR_MODE=simulation`).
4. **Camera-IP fix** (see §4.6): `CAMERA_HOST` rewritten `192.168.42.251` →
   `192.168.42.71` in place (credentials preserved).
5. **Restart** cleanly (single process on 5050): `scripts/stop.sh` +
   `pkill -9 -f vytallink.app`, verify 5050 free, `scripts/start.sh`.
6. **Validated on the Jetson:** RTSP open OK, grabber ~15 FPS, frame age 0.01–0.05 s,
   native 2304×1296; **snapshot HTTP 200, ~84 KB, 960×540**; **MJPEG ~8.3 unique
   FPS** over a 10 s window; exactly one server on 5050. (A transient diagnostic
   snapshot was fetched to a temp file to measure size/dims, then deleted — no
   footage saved.)
7. **Validated from the Mac:** the Mac's `HttpCamera` consumes the relay MJPEG at
   **8.2 FPS** with frame age p95 0.19 s and **0 reconnects** (§15).

Because the flaky Wi-Fi dropped interactive SSH mid-session, steps 4–6 were driven
by a single detached remote script (`diagnostics/jetson_cam_fix.sh`) whose redacted
result file was polled — this is how the cutover survived the link flaps.

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

## 18. Routes — what was (not) changed, and persistence

I **did not** add, remove, or persist any route — none was needed and `sudo`
requires a password on the Jetson (no non-interactive privilege).

- **Jetson → Mac:** a pre-existing temporary `/32` route pins it to wlan0
  (`192.168.42.41 dev wlan0 scope link src 192.168.42.43 metric 10`). It is **live
  but not** in the wlan0 NetworkManager connection ("Secure Wireless" has no static
  `ipv4.routes`), so a reboot/reconnect drops it. It is **not required for the
  relay** (the Mac is the client; the Jetson never initiates to the Mac for relay
  traffic), so it was left as-is.
- **Jetson → camera (`.71`):** reachable over the **default eth0 route** (the wired
  and Wi-Fi /24 segments are bridged) — verified by a successful TCP/554 connect.
  **No `/32` route needed**, and this path survives reboot on its own.

If you ever want the Jetson→Mac route to persist (cosmetic for the relay), as
`rchopra` on the Jetson (CONN = `Secure Wireless`):
```sh
sudo nmcli connection modify "Secure Wireless" +ipv4.routes "192.168.42.41/32 0.0.0.0"
sudo nmcli connection up "Secure Wireless"     # re-applies without a reboot
```
To remove the live temporary route instead: `sudo ip route del 192.168.42.41/32 dev wlan0`.
Avoid creating two equal-priority default routes; do not disable Ethernet.

## 19–21. Safety confirmations

- **Alerts stayed disabled** the entire pass: `ALERTS_ENABLED=false`, dispatcher
  built with zero providers, health `alerts.status=disabled`, 0 new alerts in the
  DB (the single stored alert predates this pass).
- **No footage stored**: `SAVE_EVENT_SNAPSHOTS/CLIPS=false` on both machines;
  `data/events`/`data/clips` empty; the soak records metrics only, never frames.
  The single Jetson validation snapshot was written to `/tmp`, measured, and
  immediately deleted.
- **No secrets exposed**: health/status/debug and logs show only
  `scheme://host:port` (no full RTSP path, no camera user/password, no bearer
  token, no absolute paths). Neither `.env` was overwritten or printed — the
  Jetson `.env` was changed only by targeted in-place edits (relay-only keys +
  the camera IP), each preceded by a timestamped `.env.bak.*` backup, and camera
  credentials were never read or echoed (the IP rewrite used `sed` on the IP
  substring only).
- **Both machines on the same commit** `50c9833`, branch `hardware-integration`,
  **not pushed and not merged**; backup tags exist on each (Mac
  `backup/pre-overnight-c020726`, Jetson `backup/pre-overnight-c2cc771`).
