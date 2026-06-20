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

## Deferred (not in this change)

- **Incident vitals snapshot** (persisting the vitals at fall-confirmation time)
  is not yet implemented — the alert/patient view is live, but a fall incident
  does not yet store a point-in-time vitals snapshot.
- **Dashboard** still renders the existing health view; the new patient-state
  fields are exposed via the API but not yet surfaced in the dashboard UI.
