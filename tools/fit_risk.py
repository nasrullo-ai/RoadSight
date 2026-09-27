"""Fit the Part B logistic combiner on dev labels and calibrate the bias for alarm F1 (SPEC section 10).

Positives: processed frames in [s - 5 s, s) before each labelled accident start s. Frames inside
accidents and around near misses are ignored. Needs at least one labelled accident.

    python tools/fit_risk.py --videos data/samples --gt data/dev_labels.json [--out weights/risk.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402
from roadsight.labels import load_gt  # noqa: E402
from roadsight.risk import CausalRiskModel  # noqa: E402
from roadsight.risk.features import FEATURES  # noqa: E402


def collect(model: CausalRiskModel, path: Path) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    model.reset({"video_id": path.name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": int(cap.get(7))})
    ts, xs = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        model.step(frame, i / fps)
        if i % model.stride == 0:
            ts.append(i / fps)
            xs.append([model.last_features[k] for k in FEATURES])
        i += 1
    cap.release()
    return np.array(ts), np.array(xs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/samples")
    ap.add_argument("--gt", default="data/dev_labels.json")
    ap.add_argument("--out", default="weights/risk.json")
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression

    gt = load_gt(args.gt)
    model = CausalRiskModel.from_config()
    X, y, per_video = [], [], []
    for vid, g in sorted(gt.items()):
        path = Path(args.videos) / vid
        if not path.exists():
            continue
        t, x = collect(model, path)
        acc = [e for e in g.get("events", []) if e[2] == "accident"]
        near = [e for e in g.get("events", []) if e[2] == "near_miss"]
        lab = np.zeros(len(t), int)
        ign = np.zeros(len(t), bool)
        for s, e, _ in acc:
            lab[(t >= s - 5) & (t < s)] = 1
            ign |= (t >= s) & (t <= e)
        for s, e, _ in near:
            ign |= (t >= s - 5) & (t <= e)
        keep = ~ign | (lab == 1)
        X.append(x[keep])
        y.append(lab[keep])
        per_video.append((vid, t, x, acc))
    X = np.concatenate(X)
    y = np.concatenate(y)
    if y.sum() == 0:
        print("no labelled accidents in the dev set -- keeping hand-set weights", file=sys.stderr)
        return 1
    clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=0, max_iter=1000).fit(X, y)
    w = {k: round(float(c), 4) for k, c in zip(FEATURES, clf.coef_[0])}

    # Bias: maximise alarm F1 (uses the same smoothing as the model via the evaluator's alarm logic).
    best = (-1.0, float(clf.intercept_[0]))
    for b in np.linspace(clf.intercept_[0] - 4, clf.intercept_[0] + 2, 61):
        pred = {}
        gtv = {}
        for vid, t, x, _ in per_video:
            z = b + x @ np.array([w[k] for k in FEATURES])
            s = 1 / (1 + np.exp(-z))
            pred[vid] = {"risk": np.stack([t, s], 1).tolist(), "events": []}
            gtv[vid] = gt[vid]
        f1 = (evaluate.evaluate_part_b(gtv, pred) or {}).get("f1_alarm", 0.0)
        if f1 > best[0]:
            best = (f1, float(b))
    w["bias"] = round(best[1], 4)
    Path(args.out).write_text(json.dumps(w, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {w} (dev alarm F1 {best[0]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
