"""Grid-search rule thresholds and boundary offsets per class against dev labels (SPEC section 9).

Perception runs once per video (cached via ROADSIGHT_CACHE_DIR); each grid point only re-runs one
rule and the post-processing for that class. Writes ``configs/tuned.yaml`` (merged automatically by
``load_config``) and a before/after report for the website.

    ROADSIGHT_CACHE_DIR=.cache python tools/tune.py --videos data/samples --gt data/dev_labels.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
from roadsight import CLASSES  # noqa: E402
from roadsight.config import deep_update, load_config  # noqa: E402
from roadsight.events.base import RULES  # noqa: E402
from roadsight.events.postprocess import postprocess  # noqa: E402
from roadsight.pipeline import EventPipeline  # noqa: E402

GRIDS = {
    "stopped_vehicle": {"min_stationary_sec": [8, 10, 15], "stop_speed": [0.08, 0.1, 0.15], "lone_min_sec": [30, 60, 120]},
    "congestion": {"speed_thr": [0.15, 0.2, 0.3], "min_vehicles": [4, 6, 8], "min_len_no_signal_sec": [30, 60, 90]},
    "jaywalking": {"edge_margin_bl": [0.1, 0.3, 0.5], "min_sec": [0.5, 1.0, 1.5]},
    "wrong_way": {"min_strength": [0.5, 0.6, 0.7], "min_sec": [1.0, 1.5, 2.0]},
    "accident": {"min_confidence": [0.5, 0.6, 0.7], "drop_ratio": [0.3, 0.35, 0.45]},
    "near_miss": {"min_confidence": [0.6, 0.7, 0.8], "ttc_thr": [0.8, 1.0, 1.2]},
}
DEFAULT_GRID = {"min_confidence": [0.5, 0.6, 0.7]}
POST_GRID = {"merge_gap": [1.0, 2.0, 4.0]}


def class_f1(pred_by_video: dict, gt: dict, label: str) -> tuple[float, float, int, int]:
    """Mean F1 over tIoU {.3,.5,.7}, precision at .3, #pred, #gt for one class."""
    f1s, prec03 = [], 0.0
    n_pred = sum(len([e for e in v if e[2] == label]) for v in pred_by_video.values())
    n_gt = sum(len([e for e in g.get("events", []) if e[2] == label]) for g in gt["videos"].values())
    for thr in evaluate.TIOU_THRESHOLDS:
        tp = 0
        for vid, g in gt["videos"].items():
            gs = [(e[0], e[1]) for e in g.get("events", []) if e[2] == label]
            ps = [(e[0], e[1]) for e in pred_by_video.get(vid, []) if e[2] == label]
            tp += evaluate.match_count(gs, ps, thr)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        if thr == 0.3:
            prec03 = p
    return float(np.mean(f1s)), prec03, n_pred, n_gt


def boundary_offsets(pred_by_video: dict, gt: dict, label: str) -> tuple[float, float]:
    ds, de = [], []
    for vid, g in gt["videos"].items():
        gs = [e for e in g.get("events", []) if e[2] == label]
        ps = [e for e in pred_by_video.get(vid, []) if e[2] == label]
        for ge in gs:
            best = max(ps, key=lambda p: evaluate.tiou((ge[0], ge[1]), (p[0], p[1])), default=None)
            if best is not None and evaluate.tiou((ge[0], ge[1]), (best[0], best[1])) >= 0.3:
                ds.append(ge[0] - best[0])
                de.append(ge[1] - best[1])
    return (float(np.median(ds)), float(np.median(de))) if ds else (0.0, 0.0)


def run_class(results: dict, label: str, ev_cfg: dict) -> dict:
    out = {}
    for vid, res in results.items():
        try:
            segs = RULES[label](ev_cfg[label], res.scene).detect(res.tracks, res.signal, res.meta)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label} failed on {vid}: {exc!r}", file=sys.stderr)
            segs = []
        out[vid] = postprocess(segs, res.meta.duration, {label: ev_cfg[label]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/samples")
    ap.add_argument("--gt", default="data/dev_labels.json")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="configs/tuned.yaml")
    ap.add_argument("--report", default="web/static/data/tuning.json")
    args = ap.parse_args(argv)
    os.environ.setdefault("ROADSIGHT_CACHE_DIR", str(ROOT / ".cache"))

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    base = load_config(args.config)
    base["overrides_file"] = None
    pipe = EventPipeline(base)
    results = {}
    for vid in sorted(gt["videos"]):
        path = Path(args.videos) / vid
        if path.exists():
            results[vid] = pipe.run_full(str(path))
    gt = {"videos": {k: v for k, v in gt["videos"].items() if k in results}}
    if not results:
        print("no dev videos found", file=sys.stderr)
        return 1

    tuned: dict = {"events": {}}
    report = {"classes": {}}
    for label in CLASSES:
        ev = base["events"].get(label, {})
        if not ev.get("enabled", False):
            continue
        before_pred = run_class(results, label, base["events"])
        f_before, p_before, n_pred0, n_gt = class_f1(before_pred, gt, label)
        best = (f_before, {}, before_pred)
        grid = {**GRIDS.get(label, DEFAULT_GRID), **POST_GRID}
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            cfg = deep_update(base["events"], {label: params})
            pred = run_class(results, label, cfg)
            f, _, _, _ = class_f1(pred, gt, label)
            if f > best[0] + 1e-9:
                best = (f, params, pred)
        f_after, params, pred = best
        cfg = deep_update(base["events"], {label: params})
        so, eo = boundary_offsets(pred, gt, label)
        if so or eo:
            trial = deep_update(cfg, {label: {"start_offset": round(so, 2), "end_offset": round(eo, 2)}})
            pred2 = run_class(results, label, trial)
            f2, _, _, _ = class_f1(pred2, gt, label)
            if f2 > f_after:
                params = {**params, "start_offset": round(so, 2), "end_offset": round(eo, 2)}
                f_after, pred = f2, pred2
        f_after, p_after, n_pred, _ = class_f1(pred, gt, label)
        enabled = not (n_pred > 0 and p_after < 0.5)
        if not enabled:
            params = {**params, "enabled": False}
        if params:
            tuned["events"][label] = params
        report["classes"][label] = {
            "f1_before": round(f_before, 3),
            "f1_after": round(f_after, 3),
            "precision@0.3": round(p_after, 3),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "enabled": enabled,
            "params": params,
        }
        print(
            f"{label:<20s} F1 {f_before:.3f} -> {f_after:.3f}  P@.3 {p_after:.2f}  pred {n_pred} gt {n_gt}  {'ON' if enabled else 'OFF'}  {params}"
        )

    Path(args.out).write_text(
        "# Written by tools/tune.py -- merged on top of configs/default.yaml\n" + yaml.safe_dump(tuned, sort_keys=True), encoding="utf-8"
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
