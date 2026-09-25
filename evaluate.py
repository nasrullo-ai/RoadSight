"""STAND-IN evaluator -- replace with the organizer's evaluate.py from the starter kit, unchanged.

This file re-implements the scoring described in SPEC.md so the team can iterate before the
starter kit is available. The metric details (Score B combination in particular) are our reading
of the task description and may differ from the official implementation.

Usage:
    python evaluate.py --pred predictions.json --validate-only
    python evaluate.py --pred predictions.json --gt data/dev_labels.json [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle", "jaywalking",
    "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
PRE_ACCIDENT_SEC = 5.0
ALARM_THRESHOLD = 0.5
ALARM_MERGE_SEC = 2.0
ALARM_WINDOW_SEC = 10.0
NEAR_MISS_IGNORE_SEC = 5.0


# ----------------------------------------------------------------------------- validation
def validate(pred: dict, gt: dict | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(pred, dict) or "videos" not in pred or not isinstance(pred["videos"], dict):
        return ["top level must be an object with a 'videos' object"]
    if not isinstance(pred.get("team", ""), str):
        errors.append("'team' must be a string")
    for vid, entry in pred["videos"].items():
        if not isinstance(entry, dict):
            errors.append(f"{vid}: entry must be an object")
            continue
        events = entry.get("events", [])
        if not isinstance(events, list):
            errors.append(f"{vid}: events must be a list")
            events = []
        duration = None
        if gt and vid in gt.get("videos", {}):
            duration = float(gt["videos"][vid].get("duration", 0)) or None
        last_end: dict[str, float] = {}
        for i, ev in enumerate(sorted(events, key=lambda e: (str(e[2]) if len(e) > 2 else "", e[0] if e else 0))):
            if not (isinstance(ev, list) and len(ev) == 3):
                errors.append(f"{vid}: event {i} must be [start, end, label]")
                continue
            s, e, lab = ev
            if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                errors.append(f"{vid}: event {ev} has non-numeric bounds")
                continue
            if not (0 <= s < e):
                errors.append(f"{vid}: event {ev} needs 0 <= start < end")
            if duration is not None and e > duration + 1e-6:
                errors.append(f"{vid}: event {ev} ends after duration {duration}")
            if lab not in CLASSES:
                errors.append(f"{vid}: unknown label {lab!r}")
            if s < last_end.get(lab, -1.0) - 1e-9:
                errors.append(f"{vid}: same-class overlap for {lab} at {s}")
            last_end[lab] = max(last_end.get(lab, -1.0), e)
        risk = entry.get("risk", [])
        if not isinstance(risk, list):
            errors.append(f"{vid}: risk must be a list")
            continue
        prev_t = -np.inf
        for r in risk:
            if not (isinstance(r, list) and len(r) == 2):
                errors.append(f"{vid}: risk entries must be [t, score]")
                break
            t, sc = r
            if not (0.0 <= float(sc) <= 1.0):
                errors.append(f"{vid}: risk score {sc} outside [0, 1]")
                break
            if t < prev_t:
                errors.append(f"{vid}: risk timestamps not sorted")
                break
            prev_t = t
    return errors


# ----------------------------------------------------------------------------- Part A
def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_count(gt: list, pred: list, thr: float) -> int:
    """Greedy one-to-one matching by highest tIoU."""
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            v = tiou(g, p)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_g, used_p, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            tp += 1
    return tp


def score_a(pred: dict, gt: dict) -> dict:
    vids = list(gt["videos"].keys())
    present = set()
    for v in vids:
        present |= {e[2] for e in gt["videos"][v].get("events", [])}
        present |= {e[2] for e in pred["videos"].get(v, {}).get("events", [])}
    per_class = {}
    for c in sorted(present):
        f1s = []
        for thr in TIOU_THRESHOLDS:
            tp = n_gt = n_pred = 0
            for v in vids:
                g = [(e[0], e[1]) for e in gt["videos"][v].get("events", []) if e[2] == c]
                p = [(e[0], e[1]) for e in pred["videos"].get(v, {}).get("events", []) if e[2] == c]
                tp += match_count(g, p, thr)
                n_gt += len(g)
                n_pred += len(p)
            prec = tp / n_pred if n_pred else 0.0
            rec = tp / n_gt if n_gt else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
        per_class[c] = float(np.mean(f1s))
    score = float(np.mean(list(per_class.values()))) if per_class else 1.0
    return {"score_a": score, "per_class_f1": per_class}


# ----------------------------------------------------------------------------- Part B
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / max(1, y.sum()))


def alarms_from(times: np.ndarray, scores: np.ndarray) -> list[float]:
    """Alarm start times: runs with score >= 0.5, runs < 2 s apart merged."""
    starts: list[float] = []
    last_on = None
    for t, sc in zip(times, scores):
        if sc >= ALARM_THRESHOLD:
            if last_on is None or t - last_on >= ALARM_MERGE_SEC:
                starts.append(float(t))
            last_on = t
    return starts


def score_b(pred: dict, gt: dict) -> dict:
    ys, ss = [], []
    n_alarm = n_alarm_tp = n_acc = n_acc_hit = 0
    tta = []
    for v, g in gt["videos"].items():
        risk = pred["videos"].get(v, {}).get("risk", [])
        if not risk:
            times = np.zeros(0)
            scores = np.zeros(0)
        else:
            arr = np.asarray(risk, dtype=np.float64)
            times, scores = arr[:, 0], arr[:, 1]
        accidents = [e for e in g.get("events", []) if e[2] == "accident"]
        near = [e for e in g.get("events", []) if e[2] == "near_miss"]
        label = np.zeros(len(times), dtype=np.int8)
        ignore = np.zeros(len(times), dtype=bool)
        for s, e, _ in accidents:
            label[(times >= s - PRE_ACCIDENT_SEC) & (times < s)] = 1
            ignore |= (times >= s) & (times <= e)
        for s, e, _ in near:
            ignore |= (times >= s - NEAR_MISS_IGNORE_SEC) & (times <= e)
        keep = ~ignore | (label == 1)
        ys.append(label[keep])
        ss.append(scores[keep])

        alarm_starts = alarms_from(times, scores)
        n_alarm += len(alarm_starts)
        used = set()
        for s, _, _ in accidents:
            n_acc += 1
            cands = [i for i, a in enumerate(alarm_starts) if s - ALARM_WINDOW_SEC <= a < s and i not in used]
            if cands:
                i = cands[0]
                used.add(i)
                n_acc_hit += 1
                tta.append(min(ALARM_WINDOW_SEC, s - alarm_starts[i]))
            else:
                tta.append(0.0)
        n_alarm_tp += len(used)

    out: dict = {}
    y = np.concatenate(ys) if ys else np.zeros(0)
    s = np.concatenate(ss) if ss else np.zeros(0)
    parts = []
    if y.sum() > 0 and len(y) > y.sum():
        ap = average_precision(y, s)
        prev = y.mean()
        ap_norm = (ap - prev) / (1 - prev)
        out["ap"] = ap
        out["ap_norm"] = ap_norm
        parts.append(max(0.0, ap_norm))
    if n_acc > 0:
        prec = n_alarm_tp / n_alarm if n_alarm else 0.0
        rec = n_acc_hit / n_acc
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out["alarm_f1"] = f1
        out["mtta_sec"] = float(np.mean(tta))
        parts += [f1, float(np.mean(tta)) / ALARM_WINDOW_SEC]
    out["n_alarms"] = n_alarm
    out["n_accidents"] = n_acc
    out["score_b"] = float(np.mean(parts)) if parts else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8")) if args.gt else None
    errors = validate(pred, gt)
    if errors:
        for e in errors[:50]:
            print(f"INVALID: {e}")
        return 1
    if args.validate_only or gt is None:
        print(f"OK: {len(pred['videos'])} videos, format valid")
        return 0

    a = score_a(pred, gt)
    b = score_b(pred, gt)
    m = 0.7 * a["score_a"] + 0.3 * b["score_b"]
    result = {"M": m, **a, **b}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score A = {a['score_a']:.4f}   Score B = {b['score_b']:.4f}   M = {m:.4f}")
        for c, f in a["per_class_f1"].items():
            print(f"  {c:<22s} F1(mean tIoU .3/.5/.7) = {f:.3f}")
        for k in ("ap", "ap_norm", "alarm_f1", "mtta_sec", "n_alarms", "n_accidents"):
            if k in b:
                print(f"  B.{k} = {b[k]:.4f}" if isinstance(b[k], float) else f"  B.{k} = {b[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
