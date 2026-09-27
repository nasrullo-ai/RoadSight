"""Record Part B's online tracker output once per video, then replay the causal risk model on it in seconds.

The detector and tracker do not depend on the risk settings, so tuning features and weights only needs the
per-frame tracks. ``record`` runs exactly what ``CausalRiskModel.step`` runs (same stride, same detector, same
tracker) and stores the tracks of every processed frame plus the first frame (for scene alignment).
``replay`` feeds them back through ``process_tracks`` and returns the per-frame risk curve the harness would write.

    python tools/risk_cache.py --videos data/samples data/dev_clips --out .cache/partb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadsight.risk import CausalRiskModel  # noqa: E402
from roadsight.scene.scene import load_aligned  # noqa: E402

VIDEO_EXT = {".mp4"}


def record(model: CausalRiskModel, video: Path, out: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = {"video_id": video.name, "fps": float(fps), "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": int(cap.get(7))}
    model.reset(meta)
    times, rows, counts, frame0 = [], [], [], None
    idx = 0
    while True:
        if idx % model.stride:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if frame0 is None:
            frame0 = frame.copy()
        tr = model.tracker.update(model._detector([frame])[0])
        times.append(idx / fps)
        rows.append(tr)
        counts.append(len(tr))
        idx += 1
    cap.release()
    path = out / f"{video.stem}.npz"
    ok, jpg = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(
        path,
        times=np.array(times),
        counts=np.array(counts),
        tracks=np.concatenate(rows) if rows else np.zeros((0, 7)),
        frame0=np.frombuffer(jpg.tobytes(), np.uint8),
        meta=np.array([meta["fps"], meta["width"], meta["height"], idx]),
        stride=model.stride,
    )
    return path


def replay(model: CausalRiskModel, path: Path, name: str) -> list[list[float]]:
    """Per-frame [t, score] exactly as the harness would record it (skipped frames repeat the last score)."""
    z = np.load(path)
    fps, w, h, n_frames = z["meta"]
    model.reset({"video_id": name, "fps": float(fps), "width": int(w), "height": int(h), "n_frames": int(n_frames)})
    frame0 = cv2.imdecode(z["frame0"], cv2.IMREAD_COLOR)
    scene, _ = load_aligned(model._scene_path, frame0, int(w), int(h), int(model.cfg.get("scene", {}).get("align_min_inliers", 40)))
    scene.auto = model.auto
    model.scene, model._needs_align = scene, False
    tracks = np.split(z["tracks"], np.cumsum(z["counts"])[:-1]) if len(z["counts"]) else []
    scores = {}
    for t, tr in zip(z["times"], tracks):
        scores[round(float(t) * fps)] = model.process_tracks(tr, float(t))
    curve, last = [], float(model.rcfg.get("floor", 0.01))
    for i in range(int(n_frames)):
        last = scores.get(i, last)
        curve.append([i / fps, last])
    return curve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["data/samples"])
    ap.add_argument("--out", default=".cache/partb")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = CausalRiskModel.from_config()
    for folder in args.videos:
        for v in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in VIDEO_EXT):
            if (out / f"{v.stem}.npz").exists():
                continue  # already recorded
            print(f"[risk_cache] {v.name}: {record(model, v, out)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
