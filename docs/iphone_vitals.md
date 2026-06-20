# iPhone vitals ↔ multi-camera incident integration

Combines iPhone-relayed vitals with per-camera fall state into one normalized,
explainable **patient state**. No RTSP URLs or credentials appear in any of these
surfaces.

> ⚠️ **The iPhone contract is new and unverified.** No prior iPhone schema existed
> in this repo (the historical "relay" is the *camera* relay). `POST /api/vitals`
> and the payload below are defined by VytalLink and must be verified against the
> real device before production use.

## Ingestion — `POST /api/vitals`

Liberal, validated JSON. At least one of `heart_rate`, `respiratory_rate`,
`motion`, `posture` is required; everything else is optional. Common aliases are
accepted (`hr`/`bpm`, `rr`/`breathing_rate`, `activity`).

```json
{
  "heart_rate": 72,            // 20–300 (plausibility guard, not medical)
  "respiratory_rate": 16,      // 3–60
  "motion": 0.1,               // 0–1 activity level
  "posture": "upright",
  "battery": 0.9,              // 0–1
  "phone_alert_score": 0.0,    // 0–1, optional phone-side score
  "device_id": "iphone-1",
  "timestamp": "2026-06-19T21:00:00Z",  // ISO-8601; optional → server time
  "sample_id": "uuid"          // optional; retries with the same id are idempotent
}
```

Validation: range + finite (NaN/Inf rejected) → `422`; bad/future/too-old
timestamp → `400`; oversized body → `413`; malformed JSON → `422`. Errors never
expose stack traces, and the server logs only a safe summary (device + which
signals were present + age) — never values or the full payload. The **source**
timestamp (`timestamp`) is stored separately from the **server-received** time.

## Normalized patient state — `GET /api/patient` (and embedded in `/latest`)

Raw source data and computed aggregates are kept distinguishable:

```json
{
  "generated_at": "...",
  "not_a_diagnosis": true,
  "vitals":    { "heart_rate", "respiratory_rate", "motion", "posture",
                 "phone_alert_score", "source", "source_timestamp", "received_at",
                 "age_seconds", "freshness" },
  "vision":    { "overall_state", "source_camera_id", "active_incident_id",
                 "person_count", "person_count_ambiguous", "cameras": { ... } },
  "freshness": { "vitals", "vitals_age_seconds", "vision" },
  "alert":     { "level", "score", "reasons": [ ... ] }
}
```

`/latest` and `/api/vitals/latest` return the same payload (one shared service
function) and keep their legacy `vital` + `simulated` top-level fields; the
`vision`/`freshness`/`alert` sections are added without breaking existing clients.

## Freshness

| vitals | camera |
|--------|--------|
| `fresh` ≤ `VITALS_FRESH_SECONDS` (15) | `fresh` ≤ `CAMERA_FRAME_FRESH_SECONDS` (5) |
| `aging` ≤ `VITALS_AGING_SECONDS` (45) | `stale` otherwise (while connected) |
| `stale` ≤ `VITALS_STALE_SECONDS` (90) | `offline` when not connected |
| `unavailable` when no sample, or older than stale | `unavailable` connected but no frame |

Old data is never treated as current; only **usable** (fresh/aging) vitals are
evaluated for abnormality, and stale/unavailable vitals are flagged — never read
as reassuring.

## Multi-camera aggregation policy

- Per-camera state is always retained.
- The overall vision state is the **worst-of the FRESH cameras only**; a stale or
  offline camera can neither mask nor erase a fresh camera's evidence.
- `source_camera_id` names the camera responsible for the aggregate state.
- `active_incident_id` is read from the DB as a separate field (the persisted
  incident from the existing fall pipeline) — one incident is never double-counted.
- Confidence values from different cameras are **never summed**.
- Fresh cameras reporting different person counts set `person_count_ambiguous`.

## Alert score (informational only)

`alert.level` ∈ `normal | info | warning | critical` with reason codes
(`fall_confirmed`, `fall_suspected`, `heart_rate_high/low`,
`respiratory_rate_high/low`, `vitals_stale`, `vitals_unavailable`,
`vision_unavailable`, `person_count_ambiguous`, `incident_active`). A confirmed
fall is `critical` even if vitals look normal; abnormal vitals without a fall are
handled independently. **The score is NOT a diagnosis and is NEVER wired to the
alert dispatcher** — only the fall-event pipeline dispatches. Thresholds are
configurable (`VITALS_HR_LOW/HIGH`, `VITALS_RR_LOW/HIGH`). Synthetic fall testing
keeps external alerts in dry-run (see docs/apple_silicon_startup.md).

## Simulator (synthetic data only)

```bash
./.venv/bin/python scripts/iphone_sim.py                    # one normal sample
./.venv/bin/python scripts/iphone_sim.py --scenario high_hr
./.venv/bin/python scripts/iphone_sim.py --scenario stale   # old timestamp -> rejected
./.venv/bin/python scripts/iphone_sim.py --count 5 --interval 2
```

Scenarios: `normal`, `high_hr`, `low_hr`, `resp`, `lying`, `minimal`, `stale`,
`duplicate`, `malformed`. Localhost by default; `--url` to target another host.

## Compatibility adapter (aliases + conflicts)

A small, documented alias set is normalized into the canonical model — it is NOT
an unbounded permissive parser. Accepted aliases:

| canonical | accepted input keys |
|-----------|---------------------|
| `heart_rate` | `heart_rate`, `hr`, `bpm` |
| `respiratory_rate` | `respiratory_rate`, `rr`, `breathing_rate`, `resp_rate`, `br` |
| `motion` | `motion`, `activity`, `activity_level` |
| `phone_alert_score` | `phone_alert_score`, `alert_score` |
| `timestamp` | `timestamp`, `time`, `ts`, `source_timestamp`, `device_timestamp`, `recorded_at` |
| `sample_id` | `sample_id`, `id` |

- **Conflicting aliases with different values are rejected** (e.g. `heart_rate:72`
  + `hr:80` → 422); identical values are fine. Units are never guessed.
  Booleans-as-numbers are rejected. Unknown keys are ignored (not nested-parsed).
- The ingest response reports `contract_form` (`canonical`/`alias`) and
  `accepted_fields`; the stored vital records `contract_form` in metadata to help
  reconcile the real device later.

## Patient-state schema version

`/api/patient` and the embedded sections carry `"version"` (currently `1`). Bump
it when the structure changes so the dashboard / future phone integrations can
adapt without guessing. Legacy top-level `vital`/`simulated` fields are unaffected.

## Incident vitals snapshot

On the **first** confirmation of a fall incident, exactly one vitals snapshot is
persisted to the `incident_vitals` table (UNIQUE on `event_uid` — one per
incident). It captures: camera id, confirmation time, latest vitals sample id,
heart/respiratory rate, posture, phone + computed alert score, alert level +
reason codes, source & received timestamps, vitals age, freshness, availability,
source (`iphone`/`simulator`/`wearable`), and a `synthetic` marker. It stores no
credentials, RTSP URLs, or raw payloads.

- One snapshot per incident; ongoing confirmed frames create no duplicates.
- A resolved-then-new fall (or an independent camera) creates a new snapshot.
- No recent vitals → snapshot still created, marked `vitals_available=false` /
  freshness `unavailable` (never invented normal values).
- Snapshot-writer failures are isolated (never crash the camera worker/observe)
  and surfaced in `/health.persistence` (`snapshot_writer`, counts).
- The snapshot is exposed on `GET /api/events/{id}` as `incident_vitals`.
- Migration is additive (schema v2), idempotent, and preserves existing events.
- **Limitation:** no cross-camera identity correlation — two cameras viewing the
  same physical area produce two independent incidents/snapshots (by design).

## Dashboard

A **Patient status** panel (fed by the canonical `/api/patient`, no competing JS
logic) shows: alert level + plain-language reason labels, heart/respiratory rate
(bpm / breaths/min), posture, vitals freshness + sample age, per-camera fall
state + freshness, source camera, active incident, and snapshot-writer health.
Freshness is colour- AND text-coded; **stale/unavailable data is shown in
amber/red, never as a reassuring green "normal"**. Dynamic values are rendered
with `textContent` (escaped); reason codes are never injected as HTML. The panel
degrades independently so one bad field cannot break the page.

## Real-device reconciliation (the contract is still unverified)

> **The local simulator verifies the VytalLink API contract. It does NOT prove
> the current real iPhone sender uses the same payload.**

To reconcile when a real payload is available:
1. Capture one real payload from the phone (developer, with consent).
2. Remove names, identifiers, and any PHI you don't need.
3. Save it as JSON and run
   `./.venv/bin/python scripts/validate_vitals_fixture.py payload.json`
   (prints field names/types/contract form — never values).
4. If fields are unmapped, add aliases to `api/schemas.VITALS_ALIASES` deliberately.
5. Add the sanitized fixture under `tests/` and re-run the suite.

## Stale-incident reconciliation

An unresolved fall incident must not pin the patient alert baseline forever. A
reconciler runs **once on startup** and on a **bounded runtime interval**
(`INCIDENT_RECONCILE_INTERVAL_SECONDS`, default 60s), resolving stale incidents
via the supported resolve path (no new alert; the incident snapshot is untouched;
history is preserved — never deleted).

Policy (per incident, by `event.source_device`):

| condition | action |
|-----------|--------|
| age ≤ `INCIDENT_STALE_SECONDS` (300) | **keep** (recent; guards brief frame delays) |
| source camera not in current config (orphaned) | **resolve** `stale_incident_timeout` |
| source camera fresh + currently in an active fall | **keep** (genuinely ongoing) |
| source camera fresh + currently normal | **resolve** `camera_recovered` |
| source camera configured but offline/stale | **keep, mark degraded** (ambiguous) |

Resolution requires **positive** evidence (orphaned, or a fresh source camera
showing normal) — a configured camera that is merely offline is *absence of
evidence*, so the incident stays open and `/health` goes degraded. Crucially, a
sustained `confirmed_fall` does not advance `updated_at`, so the decision uses the
camera's **current live state**, never age alone — an ongoing fall is never
closed. `/health.persistence.reconciliation` surfaces enabled/counts/failures/
open-ambiguous (no incident details). Config: `INCIDENT_STALE_SECONDS`,
`INCIDENT_RECONCILE_ON_STARTUP`, `INCIDENT_AUTO_RESOLVE_ENABLED`,
`INCIDENT_RECONCILE_INTERVAL_SECONDS`.

## Privacy

Only normalized vitals are stored; full request bodies are never persisted or
logged (only a safe summary: device + which signals + age). `/health` reports
status + freshness classification + safe counts — never patient vitals values.
Snapshots store no credentials/URLs/raw payloads.
