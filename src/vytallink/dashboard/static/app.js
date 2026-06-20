"use strict";
// VytalLink dashboard — vanilla JS, polls the API. No live video feed.

const POLL_MS = 3000;
const $ = (id) => document.getElementById(id);

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function setBadge(el, status) {
  const s = (status || "unknown").toLowerCase();
  el.className = "badge badge-" + (["ok", "degraded", "down", "disabled"].includes(s) ? s : "unknown");
  el.textContent = s;
}

function chip(status) {
  const s = (status || "unknown").toLowerCase();
  const cls = ["ok", "degraded", "down", "disabled"].includes(s) ? s : "unknown";
  return `<span class="chip chip-${cls}">${s}</span>`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleString(); } catch (_) { return iso; }
}
function fmtUptime(sec) {
  if (sec == null) return "—";
  sec = Math.floor(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? h + "h " : "") + (m ? m + "m " : "") + s + "s";
}
function pct(x) { return x == null ? "—" : Math.round(x * 100); }

let CONTROLS_ENABLED = false;

function renderHealth(h) {
  setBadge($("overall-badge"), h.overall);
  $("s-overall").textContent = h.overall;
  const fall = $("s-fall");
  fall.textContent = (h.fall_state || "—").replace(/_/g, " ");
  fall.className = "value fall-" + (h.fall_state || "normal");
  $("s-camera").innerHTML = chip(h.camera && h.camera.status);
  $("s-detector").innerHTML = chip(h.detector && h.detector.status);
  $("s-wearable").innerHTML = chip(h.wearable && h.wearable.status);
  // Inference device: show the actual accelerator (Apple MPS / CUDA / CPU).
  const det0 = h.detector || {};
  $("s-gpu").textContent = det0.device_label || det0.device || "—";
  $("s-db").innerHTML = chip(h.database && h.database.status);
  $("s-uptime").textContent = fmtUptime(h.uptime_seconds);
  $("version").textContent = "v" + (h.version || "?");

  // Exactly ONE top-level mode badge: SIMULATION xor LIVE (driven by the
  // single authoritative flag h.simulation.active = vision in simulation).
  const isSim = !!(h.simulation && h.simulation.active);
  $("sim-indicator").hidden = isSim ? false : true;
  $("live-indicator").hidden = isSim ? true : false;
  $("vitals-sim").hidden = !isSim;

  renderHardware(h);
  renderLiveVideo(h);

  // Warnings.
  const warnings = [];
  if (h.disk_warning) warnings.push("LOW DISK (" + (h.disk && h.disk.percent) + "%)");
  if (h.camera && h.camera.status === "down") warnings.push("CAMERA DOWN");
  if (h.wearable && h.wearable.status === "down") warnings.push("WEARABLE DOWN");
  $("warnings").innerHTML = warnings.length
    ? warnings.map((w) => `<span class="chip chip-down">${w}</span>`).join(" ")
    : `<span class="muted">No active warnings.</span>`;

  // Dev controls visibility.
  CONTROLS_ENABLED = !!(h.simulation && h.simulation.controls_enabled);
  $("dev-controls").hidden = !CONTROLS_ENABLED;
}

// --- live video -----------------------------------------------------------
// The token (when the feed is protected) lives ONLY in memory and is sent via
// the Authorization: Bearer header — never in a URL, cookie, or the page source.
let VIDEO_TOKEN = null;
let videoTimer = null;

function stopProtectedFeed() {
  if (videoTimer) { clearInterval(videoTimer); videoTimer = null; }
  const img = $("live-img");
  if (img && img.src && img.src.startsWith("blob:")) { URL.revokeObjectURL(img.src); img.removeAttribute("src"); }
}

function startProtectedFeed() {
  if (videoTimer) return;
  const img = $("live-img");
  const poll = async () => {
    try {
      const res = await fetch("/api/camera/snapshot.jpg", {
        headers: { Authorization: "Bearer " + VIDEO_TOKEN }, cache: "no-store",
      });
      if (res.status === 401) {
        VIDEO_TOKEN = null; stopProtectedFeed();
        $("video-unlock").hidden = false;
        const e = $("video-error"); e.hidden = false; e.textContent = "token rejected";
        return;
      }
      if (res.ok && img) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const old = img.src; img.src = url;
        if (old && old.startsWith("blob:")) URL.revokeObjectURL(old);
      }
    } catch (_) { /* transient; keep polling */ }
  };
  videoTimer = setInterval(poll, 200); // ~5 fps
  poll();
}

function unlockVideo() {
  const t = $("video-token-input").value.trim();
  if (!t) return;
  VIDEO_TOKEN = t;
  $("video-token-input").value = "";
  $("video-unlock").hidden = true;
  $("video-error").hidden = true;
  startProtectedFeed();
}

function renderLiveVideo(h) {
  const liveOn = !!h.live_video;
  const card = $("live-card");
  if (!card) return;
  card.hidden = !liveOn;
  const img = $("live-img");
  if (!liveOn) {                       // feature off
    stopProtectedFeed();
    if (img) img.removeAttribute("src");
    $("video-unlock").hidden = true;
    return;
  }
  if (h.video_protected) {             // token required
    if (VIDEO_TOKEN === null) {
      $("video-unlock").hidden = false;
      if (img && img.src && !img.src.startsWith("blob:")) img.removeAttribute("src");
    } else {
      $("video-unlock").hidden = true;
      startProtectedFeed();
    }
  } else {                             // open feed: smooth MJPEG via <img>
    stopProtectedFeed();
    $("video-unlock").hidden = true;
    if (img && !img.getAttribute("src")) img.setAttribute("src", "/api/camera/stream");
  }
}

function renderHardware(h) {
  const cam = h.camera || {};
  const det = h.detector || {};
  const gpu = h.gpu || {};
  const sim = !!(h.simulation && h.simulation.active);
  const mode = $("hw-mode");
  // Transport detail (e.g. "http_mjpeg") — the single LIVE/SIMULATION badge lives
  // in the top bar; this tag is just the source kind, never a duplicate badge.
  mode.textContent = h.mode || (sim ? "simulation" : "live");
  mode.className = "tag " + (sim ? "" : "tag-live");
  // Never show credentials or a model path — camera_name is already sanitized.
  $("hw-camera").textContent = h.camera_name || cam.description || "—";
  $("hw-camera-status").innerHTML = chip(cam.status);
  $("hw-camera-fps").textContent = cam.effective_fps != null ? cam.effective_fps : "—";
  $("hw-resolution").textContent = cam.resolution ? cam.resolution.join("×") : "—";
  $("hw-reconnects").textContent = cam.reconnects != null ? cam.reconnects : "—";
  $("hw-dropped").textContent = cam.frames_dropped != null ? cam.frames_dropped : "—";
  $("hw-device").textContent = det.device_label || det.device || (sim ? "n/a (sim)" : "—");
  $("hw-inf-fps").textContent = det.inference_fps != null ? det.inference_fps : "—";
  $("hw-inf-ms").textContent = det.avg_inference_ms != null ? det.avg_inference_ms + " ms" : "—";
  // Accelerator detail — shows "Apple MPS available (CUDA unavailable)" etc.
  $("hw-gpu").textContent = (gpu && gpu.detail)
    || det.device_label
    || (sim ? "n/a (sim)" : "—");
  $("hw-last-frame").textContent = fmtTime(h.latest_frame_time);
  $("hw-last-inf").textContent = fmtTime(h.latest_inference_time);
}

// --- patient state (normalized /api/patient) ------------------------------
const REASON_LABELS = {
  fall_confirmed: "Confirmed fall",
  fall_suspected: "Suspected fall",
  heart_rate_high: "High heart rate",
  heart_rate_low: "Low heart rate",
  respiratory_rate_high: "High respiratory rate",
  respiratory_rate_low: "Low respiratory rate",
  vitals_stale: "Vitals stale",
  vitals_unavailable: "Vitals unavailable",
  vision_unavailable: "Camera coverage unavailable",
  person_count_ambiguous: "Multiple-person ambiguity",
  incident_active: "Active unresolved incident",
};
function reasonLabel(code) { return REASON_LABELS[code] || String(code).replace(/_/g, " "); }

function setAlertBadge(el, level) {
  const map = { normal: "ok", info: "unknown", warning: "degraded", critical: "down" };
  const cls = map[String(level || "normal").toLowerCase()] || "unknown";
  el.className = "badge badge-" + cls;
  el.textContent = level || "normal";   // textContent: dynamic value is escaped
}
function freshChipEl(state) {
  const s = String(state || "unavailable").toLowerCase();
  const span = document.createElement("span");
  span.className = "chip fresh-" + s;
  span.textContent = s;
  return span;
}
function chipEl(status) {
  const s = String(status || "unknown").toLowerCase();
  const cls = ["ok", "degraded", "down", "disabled"].includes(s) ? s : "unknown";
  const span = document.createElement("span");
  span.className = "chip chip-" + cls;
  span.textContent = s;
  return span;
}
function shortUid(uid) { return uid ? String(uid).slice(0, 12) : "none"; }

function renderPatient(p, h) {
  if (!p) return;
  $("p-version").hidden = false;
  $("p-version").textContent = "schema v" + (p.version != null ? p.version : "?");
  $("p-synthetic").hidden = !(h && h.synthetic_detection_mode);

  const alert = p.alert || {};
  setAlertBadge($("p-alert-level"), alert.level);
  $("p-alert-score").textContent = alert.score != null ? "score " + alert.score + "/3" : "";
  const reasons = $("p-reasons");
  reasons.textContent = "";
  const codes = alert.reasons || [];
  if (!codes.length) {
    const m = document.createElement("span");
    m.className = "muted"; m.textContent = "No active reasons.";
    reasons.appendChild(m);
  } else {
    codes.forEach((r) => {
      const c = document.createElement("span");
      c.className = "chip chip-reason"; c.textContent = reasonLabel(r);  // escaped label
      reasons.appendChild(c);
    });
  }

  const v = p.vitals || {}, f = p.freshness || {};
  $("p-hr").textContent = v.heart_rate != null ? Math.round(v.heart_rate) : "—";
  $("p-rr").textContent = v.respiratory_rate != null ? Math.round(v.respiratory_rate) : "—";
  $("p-posture").textContent = v.posture || "—";
  const fr = $("p-fresh");
  fr.textContent = f.vitals || "unavailable";
  fr.className = "vital-num value-sm fresh-" + String(f.vitals || "unavailable");
  $("p-updated").textContent = fmtTime(v.source_timestamp);
  $("p-age").textContent = f.vitals_age_seconds != null ? Math.round(f.vitals_age_seconds) + " s ago" : "—";
  $("p-source").textContent = v.source || "—";

  const vis = p.vision || {};
  const vs = $("p-vision");
  vs.textContent = String(vis.overall_state || "normal").replace(/_/g, " ");
  vs.className = "value fall-" + (vis.overall_state || "normal");
  const vf = $("p-vision-fresh"); vf.textContent = ""; vf.appendChild(freshChipEl(f.vision));
  $("p-source-cam").textContent = vis.source_camera_id || "—";
  $("p-incident").textContent = vis.active_incident_id ? shortUid(vis.active_incident_id) : "none";
  const sn = $("p-snap"); sn.textContent = "";
  sn.appendChild(chipEl((h && h.persistence && h.persistence.snapshot_writer) || "unknown"));

  const cams = $("p-cameras");
  cams.textContent = "";
  Object.entries(vis.cameras || {}).forEach(([id, c]) => {
    const row = document.createElement("div"); row.className = "device";
    const left = document.createElement("span");
    const pc = c.person_count != null ? " · " + c.person_count + " person(s)" : "";
    left.textContent = id + " — " + String(c.fall_state || "normal").replace(/_/g, " ") + pc;
    row.appendChild(left);
    row.appendChild(freshChipEl(c.freshness));
    cams.appendChild(row);
  });
}

function renderStatus(s) {
  const v = s.latest_vital;
  $("v-hr").textContent = v && v.heart_rate != null ? Math.round(v.heart_rate) : "—";
  $("v-motion").textContent = v && v.motion != null ? v.motion.toFixed(2) : "—";
  $("v-battery").textContent = v && v.battery != null ? Math.round(v.battery) : "—";
  $("v-conn").textContent = v ? pct(v.connection_quality) : "—";
  $("events-count").textContent = s.counts ? s.counts.events : 0;
  $("last-update").textContent = fmtTime(s.last_update);
}

function eventCard(ev) {
  const label = ev.human_label
    ? `<span class="tag">${ev.human_label.replace(/_/g, " ")}</span>`
    : `<span class="muted">unlabeled</span>`;
  const alertTxt = ev.alert_delivered === undefined
    ? ""
    : (ev.alert_delivered ? `<span class="chip chip-ok">alert sent</span>` : `<span class="chip chip-down">alert failed</span>`);
  const conf = ev.highest_confidence != null ? Math.round(ev.highest_confidence * 100) + "%" : "—";
  const resolved = ev.state === "resolved";
  return `
    <div class="event" data-uid="${ev.event_uid}">
      <div class="event-head">
        <span class="value fall-${ev.state}">${ev.state.replace(/_/g, " ")}</span>
        ${label}
      </div>
      <div class="event-meta">
        <span><b>Start</b> ${fmtTime(ev.start_time)}</span>
        <span><b>Confidence</b> ${conf}</span>
        <span><b>Detections</b> ${ev.detection_count}</span>
        <span><b>Source</b> ${ev.source_device}</span>
      </div>
      <div class="event-actions">
        <button class="btn act-label" data-label="real_fall">Real fall</button>
        <button class="btn act-label" data-label="false_alert">False alert</button>
        <button class="btn btn-ghost act-resolve" ${resolved ? "disabled" : ""}>Resolve</button>
        ${alertTxt}
      </div>
    </div>`;
}

async function renderEvents() {
  const data = await getJSON("/api/events?limit=8");
  $("events-empty").hidden = data.total > 0;
  $("events-list").innerHTML = data.items.map(eventCard).join("");
  // Wire actions.
  document.querySelectorAll(".event").forEach((card) => {
    const uid = card.getAttribute("data-uid");
    card.querySelectorAll(".act-label").forEach((b) =>
      b.addEventListener("click", () => doLabel(uid, b.getAttribute("data-label"))));
    const rb = card.querySelector(".act-resolve");
    if (rb && !rb.disabled) rb.addEventListener("click", () => doResolve(uid));
  });
}

async function renderDevices() {
  const data = await getJSON("/api/devices");
  $("devices-list").innerHTML = data.items.map((d) => `
    <div class="device">
      <span>${d.display_name || d.device_id} <span class="dmeta">(${d.device_type})</span></span>
      <span>${chip(d.connection_status)}</span>
    </div>`).join("");
}

async function doLabel(uid, label) {
  try {
    await getJSON(`/api/events/${uid}/label`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    await renderEvents();
  } catch (e) { console.error("label failed", e); }
}

async function doResolve(uid) {
  try {
    await getJSON(`/api/events/${uid}/resolve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note: "Resolved from dashboard" }),
    });
    await refresh();
  } catch (e) { console.error("resolve failed", e); }
}

async function sim(action) {
  if (!CONTROLS_ENABLED) return;
  try {
    await getJSON(`/api/simulation/${action}`, { method: "POST" });
    await refresh();
  } catch (e) { console.error("simulation failed", e); }
}

async function refresh() {
  try {
    const [h, s] = await Promise.all([getJSON("/health"), getJSON("/api/status")]);
    renderHealth(h);
    renderStatus(s);
    // Patient panel degrades independently so one bad field can't break the page.
    try { renderPatient(await getJSON("/api/patient"), h); } catch (e) { console.error("patient render failed", e); }
    await Promise.all([renderEvents(), renderDevices()]);
    $("conn-banner").hidden = true;
  } catch (e) {
    $("conn-banner").hidden = false;
    console.error("refresh failed", e);
  }
}

function init() {
  $("btn-fall").addEventListener("click", () => sim("fall"));
  $("btn-normal").addEventListener("click", () => sim("normal"));
  $("btn-reset").addEventListener("click", () => sim("reset"));
  const unlockBtn = $("video-unlock-btn");
  if (unlockBtn) unlockBtn.addEventListener("click", unlockVideo);
  const tokenInput = $("video-token-input");
  if (tokenInput) tokenInput.addEventListener("keydown", (e) => { if (e.key === "Enter") unlockVideo(); });
  refresh();
  setInterval(refresh, POLL_MS);
}

document.addEventListener("DOMContentLoaded", init);
