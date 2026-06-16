# VytalLink — Pilot Checklist

A safe, staged process for any real-world test. **VytalLink is not an emergency
system or a medical device** — never rely on it for safety-critical monitoring.
A capable human observer must be present for any fall-staging test.

## 0. Before anything

- [ ] Everyone involved understands this is experimental and **not** a substitute
      for human supervision or an emergency call system.
- [ ] An emergency plan exists that does **not** depend on VytalLink.

## 1. Consent & privacy

- [ ] Written consent from the monitored person (and household members in view).
- [ ] Confirm what is captured: by default **no video is stored** and **no live
      feed is exposed**. Keep `SAVE_EVENT_SNAPSHOTS`/`SAVE_EVENT_CLIPS` off unless
      explicitly agreed.
- [ ] Camera framing avoids bathrooms/bedrooms unless specifically consented.
- [ ] Secrets (camera password, webhook secret) are only in `.env`, not shared.

## 2. Camera positioning

- [ ] Camera mounted to view the target area (e.g. living space floor) with good
      lighting and minimal occlusion.
- [ ] Lens/firmware time set correctly; stable mounting (no drift).
- [ ] Confirm the RTSP substream resolution/FPS is reasonable for the Jetson.

## 3. Network

- [ ] Jetson and camera on the same trusted LAN; note the Jetson IP from
      `scripts/start.sh`.
- [ ] Dashboard reachable from the caregiver's phone on that LAN.
- [ ] Because the dashboard has **no authentication yet**, keep it on a private
      network only.
- [ ] If using webhook alerts, confirm the endpoint is reachable and verifies the
      `X-VytalLink-Signature` HMAC.

## 4. Pre-flight (software)

- [ ] `scripts/diagnose.sh` → no FAIL (GPU WARN is expected until CUDA torch is
      installed).
- [ ] `scripts/smoke_test.sh` → PASS.
- [ ] Confirm `CONFIDENCE_THRESHOLD`, `FALL_CONFIRM_SECONDS`, `FALL_CLEAR_SECONDS`,
      `ALERT_COOLDOWN_SECONDS` are set to sensible values for the pilot.

## 5. Fall staging safety

- [ ] Use a **crash mat / padding**; stage falls slowly and deliberately.
- [ ] The person staging falls is able-bodied and consenting, **or** use a mannequin.
- [ ] A second person observes and can intervene at all times.
- [ ] Never stage falls near hazards (stairs, hard edges, furniture corners).

## 6. Event verification

- [ ] Stage a fall; confirm exactly **one** event appears and **one** alert is
      delivered (check the dashboard and `/api/events`).
- [ ] Verify the event's confidence, timestamps, and source device look correct.
- [ ] Confirm repeated movement during the fall does **not** create duplicate
      alerts.
- [ ] Use the dashboard to **label** the event (real fall / false alert) and
      **resolve** it.

## 7. False-alert logging

- [ ] For every false alert, label it `false_alert` and add a resolution note.
- [ ] Record context (lighting, pose, pets, etc.) to inform threshold tuning and
      future model training. Do not store footage unless consented.

## 8. Shutdown procedure

- [ ] `scripts/stop.sh` for a graceful stop.
- [ ] Confirm the process is gone and the PID file is cleared.
- [ ] If a systemd unit is installed later, use `sudo systemctl stop vytallink`.

## 9. After the pilot

- [ ] Review events/alerts in the dashboard and database.
- [ ] Note detection gaps/false positives for tuning.
- [ ] **Reminder:** VytalLink is not yet an emergency system. Do not leave anyone
      relying on it as their only line of safety.
