# RoadSight

WIUT Hackathon 2026, Computer Vision track. RoadSight watches one fixed road camera and, for each `.mp4`,
returns every traffic event as `[start_sec, end_sec, label]` (Part A, 14 classes) and a causal per-frame
accident risk score (Part B). Everything runs offline on one GPU; all weights are in this repository.

- Interface: [`solution.py`](solution.py) (`CLASSES`, `detect_events`, `RiskEstimator`); all logic lives in [`roadsight/`](roadsight/).
- Organizer files, unchanged: [`run_submission.py`](run_submission.py), [`evaluate.py`](evaluate.py), [`examples/`](examples/).
- Output on the four sample videos: [`predictions_samples.json`](predictions_samples.json).
- Website with the live demo: [`web/`](web/) (deployment below). Task statement: [`task.txt`](task.txt); design notes: [`SPEC.md`](SPEC.md).

## Install and run

Python 3.10+ and an NVIDIA GPU (CPU works, slower). Weights ship in [`weights/`](weights/); nothing is downloaded at run time.

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

Reproduce the committed sample predictions (the four organizer videos in `data/samples/`, see
[`data/samples/README.md`](data/samples/README.md)):

```bash
python run_submission.py --videos data/samples --out predictions_samples.json --team roadsight
```

Or with `make` (after `make setup`): `make run`, `make validate`, `make test`.

With Docker instead of `requirements.txt` (the same two commands run inside the container, no network):

```bash
docker build -t roadsight .
docker run --gpus all --network none -v $PWD/data/samples:/videos -v $PWD/outputs:/out roadsight
```

## Team

| Member | Role | Did what |
| --- | --- | --- |
| Nasrullo Nutfulloyev | Team lead, perception and event rules | Specification, detector and tracker integration, scene alignment, the 14 event rules |
| Gulasal Abduraximova | Risk model and evaluation | Causal risk estimator (Part B), review of the sample-video events, threshold tuning |
| Gulzoda Muhammadova | Website and demo | Website, live demo backend, EDA and annotated renders of the sample videos |

## Approach

Detector, tracker, scene map and one hand-written rule per class. Only object detection is learned;
every event decision, segment boundary and the risk score are explicit rules with thresholds in
[`configs/default.yaml`](configs/default.yaml).

| Stage | Learned or rule | Where |
| --- | --- | --- |
| Decoding | ffmpeg: multi-threaded, skips non-reference (B) frames, ~12.5 frames per second of video by timestamp, downscaled to 1920 px; full-resolution crops of the signal heads; OpenCV fall-back | `roadsight/io/video.py` |
| Detection | **Learned**: YOLO11m COCO weights, unchanged, FP16 at 960 px (YOLO11n on CPU) | `roadsight/perception/detector.py` |
| Tracking | ByteTrack (Ultralytics implementation), per-instance IDs | `roadsight/perception/tracker.py` |
| Track clean-up | Split at implausible jumps, interpolate gaps up to 1 s, drop tracks under 1 s | `roadsight/perception/tracktable.py` |
| Kinematics | Savitzky-Golay (Part A) or causal least squares (Part B); speeds in body lengths per second | `roadsight/perception/kinematics.py` |
| Scene | Hand-drawn [`configs/scene.json`](configs/scene.json) on [`configs/scene_ref.jpg`](configs/scene_ref.jpg), aligned to each video by SIFT + RANSAC homography; road, lane flow, crossings and signal queues also learned from each video's tracks | `roadsight/scene/` |
| Signal state | HSV colour of the two drawn signal heads | `roadsight/perception/signal.py` |
| Events | 14 rules, one file per class | `roadsight/events/rules/` |
| Post-processing | Confidence gate, merge, minimum length, clip, same-class non-overlap, rounding | `roadsight/events/postprocess.py` |
| Risk (Part B) | Own detector + tracker on ~8 frames per second; logistic combination of time to collision, braking, swerving, wrong way, red-light approach, pedestrian conflict and density; EMA and 1 s peak hold | `roadsight/risk/` |

Speeds are normalised by the typical vehicle size at each image row, so one threshold works near and
far from the camera. Velocity comes from the top edge of each box, which closer traffic rarely hides.
Rules only use reliable tracks (mean confidence at least 0.4, median size at least 2.5% of the frame
height). `fire_smoke` ships disabled: the colour-and-flicker heuristic is not precise enough without a
trained detector.

### Checking the events on the sample videos

There are no labels for the samples, so we reviewed our own predictions frame by frame
(`tools/review.py`: start, middle and end frames with the involved tracks highlighted). The verdicts
are in [`data/review_samples.json`](data/review_samples.json). Of 15 reviewed events, 10 were
correct, 3 unclear and 2 wrong (two pedestrians flagged as a near miss, a person standing at the kerb
flagged as jaywalking); both rules were fixed and a synthetic test was added for each. The
`failure_to_yield` segments were too short for the task's "enters the crossing to leaves the crossing"
convention and now cover the whole vehicle footprint on the zebra. This is a precision check only:
nobody watched the full videos for missed events, so recall is unknown.

## Runtime

The official harness on the four organizer samples (4K 29.97 fps XAVC H.264 4:2:2 10-bit), Windows 11,
NVIDIA RTX 3050 Laptop GPU (4 GB), Python 3.13, torch 2.6 + CUDA 12.6:

| Video | Light | Length | Part A (events) | Part B (risk) | Total / length |
| --- | --- | ---: | ---: | ---: | ---: |
| C3896.MP4 | noon | 340.3 s | 165.6 s | 464.4 s | 1.85x |
| C3897.MP4 | noon | 317.8 s | 145.8 s | 430.7 s | 1.81x |
| C3902.MP4 | dusk | 317.8 s | 145.8 s | 427.6 s | 1.80x |
| C3905.MP4 | dusk | 127.6 s | 57.6 s | 172.4 s | 1.80x |

The limit is 3x the video length. Part B is dominated by the harness's own OpenCV decode of every 4K
10-bit frame; our work there is ~20 ms per processed frame. Per-stage times are logged to stderr.

## Models, data and licences

| Component | Licence | Notes |
| --- | --- | --- |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and its COCO weights | AGPL-3.0 | `weights/yolo11{m,s,n}.pt`, downloaded unchanged from the Ultralytics v8.3.0 release |
| ByteTrack (Ultralytics implementation) | AGPL-3.0 (original MIT) | Tracking |
| [COCO](https://cocodataset.org) | CC BY 4.0 | Training data of the detector weights; we trained nothing |
| Organizer sample videos | Organizer's | Scene drawing, review, runtime; not redistributed |
| `vehicles.mp4`, `vehicles-2.mp4` | Roboflow `supervision` examples | Development only, not committed ([`data/dev_labels.json`](data/dev_labels.json): no events) |

No model was trained or fine-tuned, and no other dataset was used. Because Ultralytics is AGPL-3.0,
distributing this repository together with it falls under the AGPL.

Weight checksums (SHA-256):

```
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95  yolo11m.pt
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  yolo11s.pt
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  yolo11n.pt
```

`weights/risk.json` holds hand-set logistic weights (bias -6: no alarms on 26 minutes of normal traffic,
including the four samples); `tools/fit_risk.py` refits them once accidents are labelled.

## Determinism

Seeds are fixed in [`roadsight/utils/seed.py`](roadsight/utils/seed.py) (Python, NumPy, Torch, CUDA),
with deterministic cuDNN, `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms`. Frames are
picked by timestamp, never at random; RANSAC for scene alignment uses a fixed seed. Two runs on the same
machine give the same `predictions.json` (`tests/test_determinism.py`). The one non-deterministic
safeguard is the adaptive stride: it only engages when Part A runs slower than real time (on our 4 GB
laptop GPU it runs at ~0.45x, so it never engaged). Different GPUs or driver versions can change detections slightly.

## Tests

`make test` (pytest, about 1.5 minutes on GPU, 56 tests):

- `test_interface.py`: `CLASSES`, valid `detect_events` output, risk in [0, 1], no exceptions on bad input.
- `test_format.py`: the organizer's `run_submission.py` on two clips (no dropped events, inside the time budget), then `evaluate.py --validate-only`; metric sanity checks; the committed `predictions_samples.json` validates.
- `test_nonoverlap.py`: property test, post-processing never emits same-class overlaps or invalid bounds.
- `test_determinism.py`: two runs give identical events and risk.
- `test_causality.py`: risk for a prefix does not change when later frames are never fed.
- `test_offline.py`: the pipeline runs with sockets blocked.
- `test_rules.py`: one synthetic scenario per rule that must fire, plus quiet cases.
- `test_risk_logic.py`, `test_align.py`: risk on synthetic conflicts; scene alignment under camera shift and dusk light.

## Website and live demo

`make demo` serves the website and the demo API on http://localhost:7860 (FastAPI, `web/app.py`):
`POST /api/analyze`, `GET /api/status/{id}`, `GET /api/result/{id}`, `GET /api/video/{id}`, `GET /api/json/{id}`.
Uploads are limited to 2 minutes and 100 MB and run one at a time on CPU. `python tools/make_space.py`
assembles a Hugging Face Docker Space in `build/space/` (instructions printed at the end).

Pages: Home, Approach, Data (EDA), Results (every sample annotated, event timeline, risk curve, failure
cases), Live demo, Report, Team, Links. Refresh the sample renders with
`python tools/render.py --pred predictions_samples.json` and the EDA with `make eda`.

## Development workflow

1. Draw the scene once: `python tools/scene_editor.py --video data/samples/C3896.MP4`.
2. Label events: `python tools/annotate.py --video data/samples/<video>` writes `data/dev_labels.json` in the kit's ground-truth format.
3. Score: `python evaluate.py --pred predictions_samples.json --gt data/dev_labels.json --per-video`.
4. Tune: `make tune` grid-searches thresholds per class, switches off classes with dev precision below 0.5 and writes `configs/tuned.yaml` (merged automatically).
5. Review predictions visually: `ROADSIGHT_CACHE_DIR=.cache python tools/review.py --video data/samples/<video>`.

## Configuration reference

All thresholds live in [`configs/default.yaml`](configs/default.yaml):

- `video`: frame rate processed, decoder, adaptive stride.
- `detector`: weights, input size, confidence, kept classes.
- `tracker`: ByteTrack thresholds, gap interpolation, split and reliability limits.
- `scene`: path to `scene.json`, alignment, and the learned road, lane-flow, crossing and queue settings.
- `events.<class>`: `enabled`, `min_confidence`, post-processing (`merge_gap`, `min_len`, `start_offset`, `end_offset`) and rule parameters.
- `risk`: frame rate, input size, feature references, smoothing and the weights file.

## Repository layout

```
solution.py               the interface (implemented)
run_submission.py         organizer's harness, unchanged
evaluate.py               organizer's metric, unchanged
examples/                 organizer's example ground_truth.json and predictions.json
predictions_samples.json  our output on the four sample videos
requirements.txt          runtime dependencies (or Dockerfile)
weights/                  YOLO11 m/s/n, risk.json
configs/                  default.yaml, scene.json + scene_ref.jpg
roadsight/                io, perception, scene, events, risk, pipeline, viz, utils
tools/                    scene_editor, annotate, review, tune, fit_risk, render, eda, make_space
tests/                    pytest suite
web/                      website and FastAPI demo backend
data/                     dev_labels.json, review_samples.json; samples/ (videos, not committed)
```
