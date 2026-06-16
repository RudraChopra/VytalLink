# Model Candidate Report — VytalLink fall detection

**Date:** 2026-06-16
**Branch:** `hardware-integration`
**Author:** automated hardware-integration pass (read-only inspection of legacy code/models)

This report identifies the model the **real** legacy VytalLink fall pipeline used,
inspects every plausible candidate, and records an evidence-based selection. No
legacy file was modified; all inspection was read-only.

---

## 1. Two distinct legacy code bases were found

| Legacy dir | What it is | Model used | Fall logic |
|---|---|---|---|
| `/project/rudra/vytalinkv1` | **"Vytalink v1"** — same product as this repo (`VytalLink`) | `models/fall_detection.pt` | Custom YOLO **detection** model whose classes are `fallen/sitting/standing`, plus a temporal "DTS-lite" transition verifier |
| `/project/rudra/fall-detection` | Earlier, separate prototype ("visiage", Azure-backed) | `yolo models/yolov8n-pose.pt` | Generic COCO **pose** model + hand-written keypoint posture heuristics |

The canonical pipeline for *this* project is **`vytalinkv1`**: it shares the product
name, the same fusion/alert-score concept, and the same `configs/default.json`
structure. The `fall-detection` directory is an older pose-based experiment that
also contains a hard-coded Azure storage key (it was **not** copied and must not be).

### Evidence the canonical pipeline uses `fall_detection.pt`

* `vytalinkv1/src/fall_detection.py:24` — `DEFAULT_MODEL = .../models/fall_detection.pt`,
  loaded via `YOLO(path)`; `FALL_CLASS_INDEX = 0  # class 0 = "fallen"`.
* `vytalinkv1/src/vision_main.py:6` — *"Loads models/fall_detection.pt (YOLO11n,
  classes: fallen/sitting/standing) … Activity detection is not used in this pipeline."*
* `vytalinkv1/src/test_fall_fast.py:45-48` — the production live detector defines
  `FALL_CLS=0, SIT_CLS=1, STAND_CLS=2`, `CLASS_NAMES={0:"FALLEN",1:"SITTING",2:"STANDING"}`,
  and runs `model(frame, imgsz=416, conf=0.55)`.
* `vision_activity.py` (the only file that loads `yolov8n-pose.pt` inside v1) is an
  **activity** side-channel, explicitly *"not used"* by the fall pipeline.

---

## 2. Candidate inventory (all read-only)

All five candidate paths exist. SHA256 (full) and load results below were produced
by loading each on **CPU** with `ultralytics 8.0.190`, serially (memory-safe), with a
raw `torch.load` fallback when the architecture is newer than the installed loader.

| # | Path | Bytes | SHA256 (first 16) | Load | task | classes | Class type |
|---|---|---:|---|---|---|---|---|
| 1 | `vytalinkv1/models/fall_detection.pt` | 5,472,282 | `3f56ad30358d5c63` | **fails on 8.0.190** (`C3k2`) | detect (YOLO11n) | `fallen/sitting/standing` * | **custom fall detector** |
| 2 | `vytalinkv1/yolov8n-pose.pt` | 6,832,633 | `c6fa93dd1ee4a2c1` | ok | pose | `{0: person}` | generic COCO pose |
| 3 | `fall-detection/yolo models/yolov8n-pose.pt` | 6,828,990 | `7f80660bc2f97d66` | ok | pose | `{0: person}` | generic COCO pose |
| 4 | `/project/rudra/yolov8n-pose.pt` | 6,832,633 | `c6fa93dd1ee4a2c1` | ok | pose | `{0: person}` | generic COCO pose (**byte-identical to #2**) |
| 5 | `/project/rudra/models/best.pt` | 6,240,355 | `b73c9857d3a93a0c` | ok | detect | `{0: rudra}` | **unrelated** (single-class person detector) |

Full SHA256:
```
3f56ad30358d5c63bf8dbc0c1299cf68818c3d291dfb10c94107b94110aadd4c  fall_detection.pt
c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212  vytalinkv1/yolov8n-pose.pt
7f80660bc2f97d664d86fc9f50fd5903af392fe332c0d603fa0dd6c78bf8844c  fall-detection/.../yolov8n-pose.pt
c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212  /project/rudra/yolov8n-pose.pt
b73c9857d3a93a0c79ba4e092ceeb1fcbec6c8ff91755ad9eb358543e35f8dd1  /project/rudra/models/best.pt
```

\* `fall_detection.pt` could not be unpickled by the installed `ultralytics 8.0.190`
because it contains the **`C3k2`** module, which is a **YOLO11-specific** building block
(introduced in `ultralytics >= 8.3.0`). This is *positive confirmation* of the
`vision_main.py` claim that the model is **YOLO11n**. The class names are taken from
the legacy source (corroborated in three independent files); they will be verified by
loading the model under `ultralytics >= 8.3` in Phase 4.

---

## 3. Classification of each candidate

* **#1 `fall_detection.pt` — custom fall detector (SELECTED).** YOLO11n detection
  model, 3 posture classes, fall = class 0 `fallen`. This is the model the real
  VytalLink v1 fall pipeline loaded.
* **#2 / #4 `yolov8n-pose.pt` — generic COCO pose.** Identical files (same hash).
  Used by v1 only for the *unused* activity side-channel, and by the older
  `fall-detection` prototype. A pose model has **no fall class** — falls would have
  to be derived from keypoint heuristics. Not the canonical fall model.
* **#3 `fall-detection/.../yolov8n-pose.pt` — generic COCO pose.** A separately
  downloaded copy (different hash, 3 KB smaller). Same conclusion as #2.
* **#5 `best.pt` — unrelated.** `task=detect`, classes `{0: rudra}` — a single-class
  personal detector. Explicitly excluded per the task brief, confirmed by inspection.

---

## 4. Selection

**Selected model: `/project/rudra/vytalinkv1/models/fall_detection.pt`**
(YOLO11n detection, classes `0=fallen, 1=sitting, 2=standing`, fall class = `fallen`).

Rationale (evidence-based, not filename-based):
1. It is the model loaded by the canonical `vytalinkv1` fall pipeline
   (`fall_detection.py`, `vision_main.py`, `test_fall_fast.py`).
2. Its architecture (`C3k2` / YOLO11n) matches the legacy documentation exactly.
3. It is a **real fall detector with a `fallen` class** — so the new
   class-name-based `YoloFallDetector` reproduces v1 faithfully **without
   pretending a pose model has a fall class** (the explicit anti-pattern to avoid).
4. The new repo's detector defaults already mirror v1: `confidence_threshold=0.55`,
   `image_size=416`, `process_every_n_frames=3`.

The pose models are retained only as a documented fallback for a *future*
keypoint-heuristic detector; they are **not** selected, because the real pipeline
did not use them for fall detection.

---

## 5. What the legacy fall logic actually does (to reproduce cleanly)

The real v1 detector (`test_fall_fast.py`, "DTS-lite v3, strict mode") is **not** a
naive "class 0 ⇒ fall". It is:

1. **Per-frame YOLO** at `imgsz=416, conf=0.55`, taking the best `fallen/sitting/standing`
   confidences and the best person bbox.
2. **A temporal transition verifier ("DTS-lite")** that only treats a `fallen`
   posture as a *fall event* when it follows a recent **upright → fallen transition**
   (`min_upright_frames=3` sustained sit/stand ≥ 0.50, then `min_fallen_frames=3`
   sustained fallen ≥ 0.60, transition duration ≤ 2.5 s) — i.e. it distinguishes a
   **real fall** from someone **already lying down / slowly lying down**.
3. **Status ladder:** `no_person → normal → fallen_posture_only → possible_fall_event
   → confirmed_fall_event`. Confirm gate: posture ≥ 0.75, transition ≥ 0.50,
   speed-gate ≥ 0.70, event ≥ 0.72.
4. **Stability features:** latest-frame threaded grabber (`BUFFERSIZE=1`), `infer_every=3`,
   `FallHold` of 4 s on confirmed events, history cleared after 2 s with no detection,
   and explicit out-of-frame handling.

### Mapping into the new architecture (no redesign)

| v1 concept | New architecture home |
|---|---|
| custom `fall_detection.pt`, class 0 = `fallen` | `YoloFallDetector` (class-name based) |
| `imgsz=416`, `conf=0.55`, `infer_every=3` | `IMAGE_SIZE`, `CONFIDENCE_THRESHOLD`, `PROCESS_EVERY_N_FRAMES` (already defaulted to these) |
| latest-frame threaded grabber | live capture/inference thread (Phase 8) — off the FastAPI event loop |
| "sustained fallen ⇒ confirmed" | existing **state machine** `FALL_CONFIRM_SECONDS` / `FALL_CLEAR_SECONDS` (the default fall signal) |
| `FallHold` (anti-flicker) + dedup | existing event manager: one event + alert cooldown / duplicate suppression |
| 5 s rolling history bridging detection gaps | new live-only **`FallEvidenceSmoother`** (sustained fall reads as continuous evidence despite sparse YOLO detection; state machine untouched) |
| DTS-lite **upright→fallen transition gate** (anti "already lying") | reproduced as an **opt-in, gap-tolerant `PostureTransitionGate`** (`DETECTOR_REQUIRE_TRANSITION`, **off by default** so no real fall is missed). Posture alone can't fully separate a fall from a slow lie-down — that needs the velocity DTS (future work); see `hardware_integration_report.md` Phase 8. |

This keeps the working state machine, alerting, and dashboard intact while faithfully
carrying over v1's model, thresholds, latest-frame design, and its key false-positive
protection.
