"""Score Part B settings on recorded tracks (tools/risk_cache.py) with the official metric.

Ground truth: data/dev_labels.json (crash clips and clips without events) plus the organizer samples as
normal traffic from the test camera (no accidents). Reports the official Part B numbers and, separately, the
highest score on the samples: any score >= 0.5 there is a false alarm on the camera the test set comes from.

    python tools/tune_risk.py '{"min_age_sec": 0.5}' '{"weights": {"f_ttc": 10}}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from risk_cache import replay  # noqa: E402

import evaluate  # noqa: E402
from roadsight.labels import load_gt  # noqa: E402
from roadsight.risk import CausalRiskModel  # noqa: E402

CACHE = ROOT / ".cache" / "partb"
SAMPLES = ("C3896.MP4", "C3897.MP4", "C3902.MP4", "C3905.MP4")


def ground_truth() -> dict:
    gt = load_gt(ROOT / "data" / "dev_labels.json")
    for v in SAMPLES:
        gt[v] = {"duration": 0.0, "fps": 29.97, "events": []}
    return {k: v for k, v in gt.items() if (CACHE / f"{Path(k).stem}.npz").exists()}


def score(risk_overrides: dict) -> dict:
    model = CausalRiskModel.from_config(overrides={"risk": risk_overrides} if risk_overrides else None)
    gt = ground_truth()
    pred = {v: {"events": [], "risk": replay(model, CACHE / f"{Path(v).stem}.npz", v)} for v in gt}
    b = evaluate.evaluate_part_b(gt, pred) or {}
    per = {}
    for v in gt:
        r = np.array(pred[v]["risk"])
        acc = [e for e in gt[v]["events"] if e[2] == "accident"]
        pre = [float(r[(r[:, 0] >= s - 5) & (r[:, 0] < s), 1].max(initial=0)) for s, _, _ in acc]
        per[v] = {"max": round(float(r[:, 1].max()), 3), "pre_crash_max": [round(x, 3) for x in pre]}
    samples_max = max(per[v]["max"] for v in SAMPLES if v in per)
    return {
        "score_b": b.get("score_b"),
        "ap": b.get("ap"),
        "f1_alarm": b.get("f1_alarm"),
        "mtta": b.get("mtta_sec"),
        "alarms": b.get("n_alarms"),
        "samples_max": samples_max,
        "per_video": per,
    }


def main(argv=None) -> int:
    for arg in (argv or sys.argv[1:]) or ["{}"]:
        r = score(json.loads(arg))
        print(
            f"{arg}\n  Score B {r['score_b']:.3f}  AP {r['ap']:.3f}  F1 {r['f1_alarm']:.3f}  mTTA {r['mtta']:.2f}s  alarms {r['alarms']}"
            f"  | samples max {r['samples_max']:.3f}{'  FALSE ALARMS ON SAMPLES' if r['samples_max'] >= 0.5 else ''}"
        )
        for v, p in r["per_video"].items():
            if p["pre_crash_max"]:
                print(f"    {v:<32} pre-crash max {p['pre_crash_max']}  max {p['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
