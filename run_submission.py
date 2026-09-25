"""STAND-IN harness -- replace with the organizer's run_submission.py from the starter kit, unchanged.

Mirrors the documented contract: calls ``solution.detect_events(path)`` for Part A, then feeds
every frame to ``solution.RiskEstimator`` for Part B, and writes predictions.json.

Usage:
    python run_submission.py --videos data/samples --out outputs/predictions_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def run_video(solution, path: Path, risk_estimator) -> dict:
    t0 = time.perf_counter()
    try:
        events = solution.detect_events(str(path))
    except Exception:
        traceback.print_exc()
        events = []
    t_a = time.perf_counter() - t0

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1.0 <= fps <= 240.0:
        fps = 25.0
    meta = {
        "video_id": path.name,
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    risk = []
    t1 = time.perf_counter()
    try:
        risk_estimator.reset(meta)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            score = float(risk_estimator.step(frame, t))
            risk.append([round(t, 2), round(min(1.0, max(0.0, score)), 4)])
            idx += 1
    except Exception:
        traceback.print_exc()
    finally:
        cap.release()
    t_b = time.perf_counter() - t1
    duration = len(risk) / fps if risk else 0.0
    rtf = (t_a + t_b) / duration if duration else 0.0
    print(f"[harness] {path.name}: {len(events)} events, {len(risk)} risk steps, A={t_a:.1f}s B={t_b:.1f}s, "
          f"video={duration:.1f}s, total/duration={rtf:.2f}x", file=sys.stderr)
    return {"events": events, "risk": risk}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with .mp4 files (or a single file)")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--team", default="roadsight")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solution

    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT)
    estimator = solution.RiskEstimator()
    out = {"team": args.team, "videos": {}}
    for path in videos:
        out["videos"][path.name] = run_video(solution, path, estimator)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"[harness] wrote {args.out} ({len(videos)} videos)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
