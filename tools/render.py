"""Render every sample: annotated H.264 video, timeline PNG and a JSON bundle for the website.

python tools/render.py --videos data/samples --out web/static/media [--gt data/dev_labels.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadsight.pipeline import EventPipeline  # noqa: E402
from roadsight.risk import CausalRiskModel, run_risk  # noqa: E402
from roadsight.viz import CLS_NAMES, render_video, timeline_png  # noqa: E402


def result_bundle(result, risk, gt_events, timing) -> dict:
    df = result.tracks.df
    counts = df.groupby("cls")["track_id"].nunique().to_dict() if len(df) else {}
    return {
        "meta": result.meta.to_dict(),
        "events": result.events,
        "gt_events": gt_events,
        "risk": risk,
        "tracks": {CLS_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()},
        "candidates": [[round(s.start, 2), round(s.end, 2), s.label, round(s.confidence, 2)] for s in result.segments],
        "signal": result.signal.to_json(),
        "stride": result.stride,
        "timing": timing,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/samples")
    ap.add_argument("--out", default="web/static/media")
    ap.add_argument("--gt", default="data/dev_labels.json")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    if args.gt and Path(args.gt).exists():
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8")).get("videos", {})
    pipe = EventPipeline.from_config(args.config)
    risk_model = CausalRiskModel.from_config(args.config)
    index = []
    for path in sorted(Path(args.videos).glob("*.mp4")):
        t0 = time.perf_counter()
        result = pipe.run_full(str(path))
        t_a = time.perf_counter() - t0
        t1 = time.perf_counter()
        risk = run_risk(str(path), risk_model)
        t_b = time.perf_counter() - t1
        gt_events = gt.get(path.name, {}).get("events", [])
        timing = {
            "part_a_sec": round(t_a, 2),
            "part_b_sec": round(t_b, 2),
            "rtf": round((t_a + t_b) / max(result.meta.duration, 1e-6), 3),
            **result.timing,
        }
        stem = path.stem
        render_video(str(path), result, risk, out / f"{stem}.mp4", width=args.width)
        timeline_png(result.events, risk, result.meta.duration, out / f"{stem}_timeline.png", gt_events)
        bundle = result_bundle(result, risk, gt_events, timing)
        (out / f"{stem}.json").write_text(json.dumps(bundle), encoding="utf-8")
        index.append(
            {
                "video": path.name,
                "stem": stem,
                "duration": round(result.meta.duration, 2),
                "events": len(result.events),
                "rtf": timing["rtf"],
            }
        )
        print(f"[render] {path.name}: {len(result.events)} events, rtf {timing['rtf']}", file=sys.stderr)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
