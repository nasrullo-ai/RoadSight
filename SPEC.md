# RoadSight: technical specification

Markdown copy of `RoadSight_ _Technical_Specification_Traffic_Event_Detection_ _Ac.pdf` (Sep 25, 2026, @Nasrullo).
Where the implementation deviates, the deviation is noted in *italics*.

## 1. Purpose and scope

Build an offline system that reads one fixed CCTV road-camera `.mp4`, returns every traffic event as
`[start_sec, end_sec, label]` (Part A), and emits a causal per-frame accident risk score (Part B).
Deadline: Sunday 27 Sep 2026, 23:59 Tashkent time; the tagged commit at that moment is what gets run.

Deliverables:

1. Repository: code, weights, `solution.py` at the root, one-command entry point, `requirements.txt` or `Dockerfile`.
2. Public website: team, approach, EDA, annotated sample videos, live upload demo, report, links.
3. Technical report: one page on the website: what was built, what worked, what did not.

Scoring: `M = 0.7 * Score_A + 0.3 * Score_B`; `Elimination = 0.6 * M + 0.25 * Website + 0.15 * Code`.

| Component | Weight in final | Design implication |
| --- | --- | --- |
| Score A (macro F1 over tIoU 0.3/0.5/0.7) | 42% | Tight boundaries; never predict a class you cannot detect reliably |
| Score B (AP, alarm F1, mTTA) | 18% | Cheap TTC-based risk; calibrate 0.5 threshold |
| Website | 25% | Live demo is 30% of this; must not crash |
| Code | 15% | Runs on a clean machine first try (40% of this) |

Success criteria:

- `python run_submission.py` completes on all samples with zero crashes and within 3x video duration (target 1x).
- `python evaluate.py --pred predictions.json --validate-only` passes.
- Score A on our own dev labels >= 0.35; Score B > 0.
- Two consecutive runs produce identical `predictions.json`.
- Website demo processes a 2-minute upload end-to-end without errors.

## 2. Hard constraints

| Area | Constraint | How we satisfy it |
| --- | --- | --- |
| Hardware | 1 GPU, 16 GB VRAM (T4-class), 8 CPU cores, 32 GB RAM | FP16 YOLO at 640 px; batch 8-16 frames; CPU fallback path. *Part A uses 960 px (more small vehicles), Part B 640 px.* |
| Time | Part A + Part B <= 3x video duration per video | Budget 1x; shared detection cache between A and B |
| Network | No internet at evaluation | All weights in repo; `YOLO_OFFLINE=1`, no auto-downloads. *Weights committed directly (largest 41 MB), not via LFS.* |
| Weights | <= 5 GB total | YOLO11m (~40 MB) + optional small classifier; no VLM |
| Python | 3.10+ | Pin versions in `requirements.txt`; Dockerfile based on CUDA 12.1 runtime |
| Models | Open weights only; no hosted APIs at inference | Ultralytics YOLO, ByteTrack; list licences |
| Data | Public datasets + own annotations of samples only | List datasets and licences in README |
| Interface | `run_submission.py`, `evaluate.py` unchanged | Never edit them; CI test imports them as-is |
| Causality | `RiskEstimator.step` sees only past frames; no reuse of Part A output | Separate state object; no file access inside step |
| Determinism | Two runs -> same `predictions.json` | Fixed seeds, `cudnn.deterministic=True`, deterministic tracker, no random frame sampling |
| Crash handling | Crash = empty prediction for that video | Wrap internals in try/except; degrade to partial results, never raise |
| Same-class overlap | Forbidden; harness drops them | Final merge step guarantees non-overlap per class |

Licence note: Ultralytics YOLO is AGPL-3.0. Acceptable for an open competition repo; state it in the README.
If AGPL is a problem, use RT-DETR (Apache-2.0).

## 3. Interface contract

`solution.py` at the repo root is a thin adapter; all logic lives in the `roadsight/` package. It exposes
`CLASSES` (14 labels), `detect_events(video_path) -> [[float, float, str], ...]` (lazy singleton pipeline,
never raises, returns `[]` on failure) and `RiskEstimator` with `reset(meta)` and `step(frame, t_sec) -> float`
(returns the last score on internal errors).

Output rules for `detect_events`:

- `0 <= start_sec < end_sec <= duration`, floats, rounded to 2 decimals.
- `label` in `CLASSES`.
- No two segments of the same class overlap (merge before returning). Different classes may overlap.
- Empty video or no events -> `[]`. Events running past the end -> `end_sec = duration`.

Output rules for `RiskEstimator.step`:

- Input frame: BGR uint8 (H, W, 3); `t_sec`: timestamp. `meta`: video_id, fps, width, height, n_frames.
- Returns a float in [0, 1] = P(accident starts within 5 s).
- May process every Nth frame and return the cached score otherwise.
- Must not open the video, read disk caches from Part A, or look ahead.

`predictions.json` (written by the harness):
`{"team": "roadsight", "videos": {"test_001.mp4": {"events": [[12.4, 18.9, "accident"]], "risk": [[0.00, 0.01], [0.04, 0.01]]}}}`

## 4. Event classes and priorities

A wrongly predicted class that never occurs in the test set adds a 0 to the macro average, so each class
ships only if it passes a precision gate on our dev set. Tier 1 first; Tier 3 only if time allows and precision is high.

| Tier | Label | Start | End | Primary signal |
| --- | --- | --- | --- | --- |
| 1 | stopped_vehicle | Vehicle stops | Moves again / removed | Track speed ~0 for >= 10 s, outside signal queue zone |
| 1 | congestion | Queue stops moving | Queue clears | Mean speed per direction below threshold across all lanes |
| 1 | jaywalking | Pedestrian steps onto road | Leaves road | Person foot point in carriageway polygon, outside crosswalk |
| 1 | wrong_way | Enters opposing lane | Returns / leaves frame | Track heading vs lane flow field |
| 1 | accident | First visible contact | Objects stop / leave | Box overlap + abrupt deceleration + post-event stop |
| 2 | near_miss | Evasive action onset | Users clear | Low TTC + hard braking / swerve, no contact |
| 2 | red_light | Front crosses stop line on red | Leaves intersection | Traffic-light state + stop-line crossing |
| 2 | stop_line | Stops past line on red | Signal turns green | Stationary beyond stop line during red |
| 2 | failure_to_yield | Vehicle enters crossing | Leaves crossing | Vehicle in crosswalk polygon while person on it |
| 2 | solid_line_crossing | Wheel crosses line | Fully in new lane | Track crosses solid-line polyline |
| 2 | illegal_u_turn | Starts turning | Completes turn | Heading change ~180 deg in prohibited zone |
| 2 | illegal_turn | Starts turning | Completes turn | Entry lane -> exit zone not in allowed-movement table |
| 3 | road_obstacle | Obstacle appears | Removed | Static foreground blob on carriageway, not a vehicle |
| 3 | fire_smoke | First visible smoke | Clears | Fire/smoke detector or colour+motion heuristic |

Class gating: each class has `enabled` and `min_confidence` in `configs/default.yaml`. Disable a class if dev-set
precision < 0.5, unless it has zero dev occurrences and its rule is conservative. The test set contains events not in
the samples, so conservative rules for Tier 2/3 are still worth shipping.

## 5. System architecture

Detector + tracker + scene map + rule engine. Learned parts: object detection (optionally a fire/smoke classifier).
Rule-based parts: every event decision, segment boundaries and the risk score.

Part A: video -> frame reader (stride 2-3) -> detector (YOLO FP16) -> tracker (ByteTrack) -> kinematics
(speed, heading, accel) -> rule engine (14 detectors, with scene config and signal state) -> post-process
(merge, pad, NMS) -> events. Part A reads the whole file and may use smoothing and look-ahead.

Part B: frame from harness -> every Nth frame: detector + tracker (online state) -> risk features (TTC, brake,
wrong-way, pedestrian) -> logistic combiner + EMA smoothing -> score 0-1; other frames return the cached score.

Key decisions: the scene config is hand-annotated once (single fixed camera); lane directions are verified from the
dominant flow of tracks (*and used as a fall-back when no lanes are drawn*); no VLM in the scored pipeline; the shared
perception cache is inside Part A only; Part B must not read it.

## 6. Repository layout

One package, small modules, one responsibility each; no notebooks as the only source. See `README.md` for the tree.
`EventRule(cfg, scene).detect(tracks, signal, meta) -> list[Segment]`; `TrackTable` is columnar with track_id, t, cls,
boxes, foot point, speed, heading, accel, conf for every processed frame (*lane ids are computed by the rules that need them*).

## 7. Perception layer

- Frame reader: OpenCV, sequential decode, stride 2 at 25 fps (12.5 Hz), timestamp = frame_idx / fps.
- Detector: YOLO11m or YOLOv8m COCO weights, FP16, batch 16; classes person, bicycle, car, motorcycle, bus, truck.
  Confidence 0.25 for tracking input, NMS IoU 0.5. *We feed boxes from 0.1 to ByteTrack's low-score association and
  start tracks at 0.3.* TensorRT optional; `.pt` is the guaranteed default.
- Tracker: ByteTrack, `track_buffer` = 30 processed frames, deterministic. Drop tracks shorter than 1 s; interpolate
  gaps <= 1 s. Foot point = bottom-centre of box.
- Kinematics: Savitzky-Golay (Part A) or causal smoothing (Part B). Speed normalised by a per-row scale from the median
  vehicle box size ("body lengths per second"). Heading = atan2 of velocity; acceleration = derivative of speed.
  *Velocity uses the box top, which is rarely occluded; border-cut boxes are excluded from kinematics.*
- Scene (`configs/scene.json`, drawn with `tools/scene_editor.py`): carriageway, lanes with direction, stop lines with
  signal id, crosswalks, solid lines, no-U-turn zones, allowed movements, signal ROIs, queue zones, exit zones.
- Signal state: per ROI HSV thresholds for red/amber/green, brightest lamp, 1 s median filter; if no signal head is
  visible, `red_light` and `stop_line` stay disabled.

## 8. Per-class event rules (Part A)

Each rule is a pure function of TrackTable, SignalTimeline and Scene; thresholds are config values tuned on dev labels.

- stopped_vehicle: speed < 0.1 BL/s for >= 10 s on the carriageway, not in a queue zone on red, not part of
  congestion; end at speed > 0.3 BL/s for 1 s or track lost; link stationary IDs at the same place across gaps <= 5 s.
- congestion: per direction, median speed < 0.2 BL/s and vehicle count >= k for >= 15 s; must persist through a green
  phase when the signal is known; one segment per contiguous period.
- jaywalking: person foot point inside the carriageway and outside every crosswalk (1 m buffer) for >= 1 s; merge gaps
  < 2 s; ignore people next to a stopped vehicle unless they walk > 2 BL away.
- wrong_way: velocity . lane direction < -0.5 for >= 1.5 s at > 0.5 BL/s; reversing under one vehicle length does not count.
- accident: box IoU > 0.05 with |decel| > a_crash (or heading jump) within 0.5 s; confirmed by an object stationary
  >= 3 s afterwards or a person appearing; start = first contact; end = objects stationary or out of frame (cap 60 s).
- near_miss: pair TTC < 1.0 s with hard braking or a swerve (> 25 deg in 1 s) and no contact; start = evasive action;
  end = distance increasing and TTC > 3 s; suppressed if an accident is confirmed for the pair.
- red_light: vehicle front crosses the stop line on red (0.5 s grace); end at the exit zone or frame edge; optional
  right-turn-on-red exception.
- stop_line: stops (< 0.1 BL/s for 2 s) past the line but not in the intersection during red; ends at green.
- failure_to_yield: vehicle enters a crosswalk while a person is on it or about to step on it.
- solid_line_crossing: wheel line crosses a solid-line polyline and the vehicle ends on the other side.
- illegal_u_turn / illegal_turn: cumulative heading change > 150 deg in a no-U-turn zone; entry -> exit not in
  `allowed_movements`.
- road_obstacle: static foreground blob on the carriageway for > 5 s not explained by a track; also COCO animals or
  luggage on the carriageway. *Blob path implemented but off by default.*
- fire_smoke: fire/smoke detector or colour/flicker heuristic, >= 3 s, very high precision gate; *disabled by default*.

## 9. Segment post-processing

Per class: gap merge (default 2 s; 5 s congestion), minimum duration (1 s; 10 s stopped_vehicle; 15 s congestion),
learned boundary offsets, clip to [0, duration], same-class non-overlap, confidence cut, round to 2 decimals and sort.
*The confidence cut runs first so weak candidates never stretch confident ones.* `tools/tune.py` grid-searches
merge gaps, minimum lengths, offsets and main thresholds per class against dev labels.

## 10. Part B: causal risk estimator

Positives are frames in [s - 5 s, s) before each accident start; frames inside accidents and around near misses are
ignored. AP is chance-normalised. An alarm is a run with score >= 0.5 (runs < 2 s apart merge) that matches if it
starts in [s - 10 s, s). mTTA rewards early alarms (up to 10 s).

Detector + tracker on every 3rd frame; ring buffer of the last 3 s of track states; lazy-load the detector on the first
reset; reset the tracker every reset. Features: `f_ttc = exp(-TTC / 1.5 s)`, braking, swerve, wrong-way, red-light
runner, pedestrian conflict, density (*plus `f_conflict = f_ttc x max(f_brake, f_swerve)`*). Combiner
`r = sigmoid(w0 + sum w_i f_i)`, causal EMA (alpha 0.3), 1 s peak hold, density-dependent floor; weights hand-set,
then refit with logistic regression; bias chosen for alarm F1. Tests: causality (prefix scores unchanged) and speed
(<= 10 ms mean per step on a T4 at stride 3).

## 11. Runtime budget

Target <= 1.0x video duration (limit 3x). Load weights once per process and warm up; adaptive stride if the projected
time exceeds 1.5x; CPU fallback with YOLO nano (stride 5; Part B stride 6); per-stage timing to stderr and a timing
table in the README.

## 12. Dev set and evaluation workflow

`tools/annotate.py` labels events into `data/dev_labels.json` (duration, fps, events), following the start/end
conventions of section 4. Commands: `make run`, `make eval`, `python evaluate.py --pred ... --validate-only`.
Rare classes are validated on public clips and hand-made edge cases; keep a per-class error log for the website.

## 13. Website and live demo

Hosted on Hugging Face Spaces (FastAPI + static frontend, free CPU tier), static pages optionally on GitHub Pages or
Vercel. Pages: Home, Team, Approach, EDA, Results, Demo, Report, Links. Demo backend: `POST /analyze` -> job id,
`GET /status/{id}` -> progress, `GET /result/{id}` -> events, risk, annotated video URL; same `solution.py` code on CPU;
one job at a time; validate container, duration <= 120 s and size; friendly errors, never a stack trace; H.264 output.
Frontend: timeline and risk curve, click to seek; responsive; dark/light theme.

## 14. Code quality and reproducibility

Required tests: `test_interface.py`, `test_format.py`, `test_nonoverlap.py`, `test_determinism.py`,
`test_causality.py`, `test_offline.py`. Clean-machine check with `docker run --gpus all --network none`. The README
lists one-command setup and run, hardware tested, timing table, every dataset and model with licence, how weights were
produced, learned vs rule-based summary and a config reference. Python 3.10, type hints, ruff + black, seeds fixed in
`roadsight/utils/seed.py`. Tag the final commit `v1.0` before the deadline.

## 15. Milestones

| When | Milestone | Done when |
| --- | --- | --- |
| Fri 25 Sep, evening | M0 skeleton | Repo layout, `solution.py`, harness + validate pass, Dockerfile builds |
| Sat 26 Sep, morning | M1 perception | Detector + tracker + kinematics on all samples; `scene.json` drawn; EDA output |
| Sat 26 Sep, afternoon | M2 Tier 1 rules | Tier 1 rules; dev labels done; first scores |
| Sat 26 Sep, night | M3 Part B + website v1 | Risk estimator passing causality test; demo live; tag v0.1 |
| Sun 27 Sep, morning | M4 Tier 2 rules + tuning | Tier 2 rules behind gates; `tune.py` run; annotated renders |
| Sun 27 Sep, afternoon | M5 polish | Report, ablations, team page, README; clean-machine Docker test |
| Sun 27 Sep, before 22:00 | M6 freeze | Final tag v1.0 pushed |

Open questions: is a traffic signal head visible in the camera; exact CLI flags of the starter kit's
`run_submission.py` and `evaluate.py`; local rules for turns on red and U-turns.
