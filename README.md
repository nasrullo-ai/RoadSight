# RoadSight

Offline traffic event detection for one fixed CCTV road camera. For each `.mp4` it returns every
traffic event as `[start_sec, end_sec, label]` (Part A, 14 classes) and a causal per-frame accident
risk score (Part B). Everything runs locally on one GPU with no network access.

- Interface: [`solution.py`](solution.py) (`detect_events`, `RiskEstimator`); all logic lives in [`roadsight/`](roadsight/).
- Specification: [`SPEC.md`](SPEC.md). Website and live demo: [`web/`](web/).

## Run it

Python 3.10+ and an NVIDIA GPU (CPU works, slower):

```bash
pip install -r requirements.txt
python run_submission.py --videos data/samples --out outputs/predictions_samples.json
python evaluate.py --pred outputs/predictions_samples.json --validate-only
```

Or with `make` (after `make setup`): `make run`, `make validate`, `make eval`, `make test`.

Clean-machine check without the network:

```bash
docker build -t roadsight .
docker run --gpus all --network none -v $PWD/data/samples:/videos -v $PWD/outputs:/out roadsight
```

> `run_submission.py` and `evaluate.py` in this repository are **stand-ins** written from the task
> description, so the pipeline can be exercised before the starter kit arrives. Replace them with the
> organizer's files unchanged; nothing in `roadsight/` depends on them.

## Hardware tested and timing

Tested on Windows 11, Python 3.12, NVIDIA RTX 4050 Laptop GPU (6 GB), 16 CPU threads. Dependencies are
pinned to versions that also ship wheels for Python 3.10; the Docker image uses Python 3.10 with CUDA 12.1.

| Clip | Resolution | Length | Part A (events) | Part B (risk) | Total / length |
| --- | --- | ---: | ---: | ---: | ---: |
| vehicles-2.mp4 (city underpass) | 1920x1080 | 42.5 s | 18.2 s | 12.0 s | 0.71x |
| vehicles.mp4 (highway) | 3840x2160 | 21.5 s | 9.1 s | 11.8 s | 0.97x |

The limit is 3x the video length. Part A detects every 2nd frame at 960 px (YOLO11m, FP16, batch 16);
Part B detects every 3rd frame at 640 px and averages 5 to 9 ms per `step` call. The stride rises
automatically only if Part A alone runs slower than real time. Per-stage times are logged to stderr.

## How it works

| Stage | Learned or rule | Where |
| --- | --- | --- |
| Decoding | ffmpeg: multi-threaded, skips non-reference (B) frames, picks ~12.5 Hz by timestamp, downscales to 1920 px; OpenCV fall-back | `roadsight/io/video.py` |
| Detection | Learned: YOLO11m COCO weights, unchanged (YOLO11n on CPU) | `roadsight/perception/detector.py` |
| Tracking | ByteTrack (Ultralytics implementation), per-instance IDs | `roadsight/perception/tracker.py` |
| Track clean-up | Split at implausible jumps, interpolate gaps up to 1 s, drop tracks under 1 s | `roadsight/perception/tracktable.py` |
| Kinematics | Savitzky-Golay (Part A) or least squares (Part B); speeds in body lengths per second | `roadsight/perception/kinematics.py` |
| Scene | Hand-drawn `configs/scene.json` on `configs/scene_ref.jpg`, aligned to each video by SIFT + RANSAC homography; road, lane flow, crosswalks and queue zones learned from tracks as fall-back | `roadsight/scene/` |
| Signal state | HSV colour of drawn signal-head boxes | `roadsight/perception/signal.py` |
| Events | 14 rules, one file per class | `roadsight/events/rules/` |
| Post-processing | Confidence gate, merge, minimum length, offsets, clip, same-class non-overlap, rounding | `roadsight/events/postprocess.py` |
| Risk | Logistic combination of time to collision, braking, swerving, wrong way, red-light approach, pedestrian conflict and density; EMA and 1 s peak hold | `roadsight/risk/` |

Speeds are normalised by the typical vehicle size at each image row (fitted per video), so one
threshold works near and far from the camera. Velocity is taken from the top edge of each box, which
closer traffic rarely hides. Rules only use "reliable" tracks (mean confidence at least 0.4, median size
at least 2.5% of the frame height). Near-miss and risk thresholds scale with each video's own noise.

Classes that need geometry (red light, stop line, solid line, U-turn and turn zones) switch themselves
off until `scene.json` provides it. `fire_smoke` ships disabled: the colour-and-flicker heuristic is
not precise enough without a trained detector.

## Adapting to the competition camera

1. Put the organizer samples in `data/samples/`.
2. Draw the scene once: `python tools/scene_editor.py --video data/samples/<sample>.mp4` (carriageway, lanes with
   direction, stop lines, signal boxes, crosswalks, solid lines, zones). Coordinates are stored normalised.
3. Label events: `python tools/annotate.py --video data/samples/<sample>.mp4` writes `data/dev_labels.json`.
4. Tune: `make tune` grid-searches thresholds and boundary offsets per class, switches off classes whose dev
   precision is below 0.5 and writes `configs/tuned.yaml` (merged automatically) plus a report for the website.
5. Fit risk weights when accidents are labelled: `python tools/fit_risk.py` writes `weights/risk.json`.
6. Refresh the website: `make render eda`, then `python tools/make_space.py` to build the demo Space.

## Tests

`make test` (pytest, about 1 minute on GPU) covers:

- `test_interface.py`: `CLASSES`, valid `detect_events` output, risk scores in [0, 1], no exceptions on bad input.
- `test_format.py`: harness on two clips, then `evaluate.py --validate-only`; metric sanity checks.
- `test_nonoverlap.py`: property test, post-processing never emits same-class overlaps or invalid bounds.
- `test_determinism.py`: two runs give identical events and risk.
- `test_causality.py`: risk for a prefix does not change when the video is truncated.
- `test_offline.py`: the pipeline runs with sockets blocked.
- `test_rules.py`: one synthetic scenario per rule that must fire, plus quiet cases.

## Website and live demo

`make demo` serves the site and the demo API on http://localhost:7860 (FastAPI, `web/app.py`):
`POST /api/analyze`, `GET /api/status/{id}`, `GET /api/result/{id}`, `GET /api/video/{id}`, `GET /api/json/{id}`.
Uploads are limited to 2 minutes and 100 MB and processed one at a time. `python tools/make_space.py`
assembles a CPU Hugging Face Docker Space in `build/space/`. The static pages can also be hosted on
GitHub Pages or Vercel; set the API base in `web/static/js/config.js`.

Before publishing, fill in `web/static/data/team.json` (names, roles, photos, links) and
`web/static/data/links.json` (repository URL), and replace the development clips in
`web/static/media` with renders of the organizer samples (`make render`).

## Models, data and licences

| Component | Licence | Notes |
| --- | --- | --- |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 8.3.40 and its COCO weights | AGPL-3.0 | `weights/yolo11{m,s,n}.pt`, downloaded unchanged from the Ultralytics v8.3.0 release |
| ByteTrack (Ultralytics implementation) | AGPL-3.0 (original MIT) | Tracking |
| [COCO](https://cocodataset.org) | CC BY 4.0 | Training data of the detector weights |
| Development clips `vehicles.mp4`, `vehicles-2.mp4` | Roboflow `supervision` examples | Local testing only, not committed |

Because Ultralytics is AGPL-3.0, distributing this repository together with it falls under the AGPL.
If that is a problem, swap the detector for RT-DETR (Apache-2.0).

Weight checksums (SHA-256):

```
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95  yolo11m.pt
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  yolo11s.pt
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  yolo11n.pt
```

No weights were trained. `weights/risk.json` holds hand-set logistic weights; `tools/fit_risk.py`
refits them on dev labels (seeded). `outputs/predictions_samples.json` is reproduced by the
`run_submission.py` command above; runs are deterministic (fixed seeds, deterministic cuDNN).

## Configuration reference

All thresholds live in [`configs/default.yaml`](configs/default.yaml):

- `video`: processing stride, adaptive stride trigger.
- `detector`: weights, input size, confidence, kept classes.
- `tracker`: ByteTrack thresholds, gap interpolation, split and reliability limits.
- `scene`: path to `scene.json` and the automatic road, lane-flow and crosswalk settings.
- `events.<class>`: `enabled`, `min_confidence`, post-processing (`merge_gap`, `min_len`, `start_offset`, `end_offset`) and rule parameters.
- `risk`: stride, input size, feature references, smoothing and the weights file.

## Repository layout

```
solution.py            interface adapter
run_submission.py      stand-in harness (replace with the organizer's)
evaluate.py            stand-in evaluator (replace with the organizer's)
configs/               default.yaml, scene.json, tuned.yaml (from tools/tune.py)
weights/               YOLO11 m/s/n, risk.json
roadsight/             io, perception, scene, events, risk, pipeline, viz, utils
tools/                 scene_editor, annotate, tune, fit_risk, render, eda, make_space
tests/                 pytest suite
web/                   FastAPI demo backend and static website
data/                  dev_labels.json; samples/ (not committed)
```
