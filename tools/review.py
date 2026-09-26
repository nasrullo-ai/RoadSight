"""Contact sheets for reviewing detections: start / middle / end frames of every event (and optionally the
strongest candidates below the confidence gate), with the involved tracks in red and the scene overlay.

    ROADSIGHT_CACHE_DIR=.cache python tools/review.py --video data/samples/C3896.MP4 --out outputs/review [--candidates 0.4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadsight.pipeline import EventPipeline  # noqa: E402

WIDTH = 960


def grab(video: str, t: float, size: tuple[int, int]) -> np.ndarray | None:
    """One frame at ``t`` seconds via ffmpeg (fast keyframe seek, then exact decode)."""
    import imageio_ffmpeg

    w, h = size
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(t, 0):.3f}",
        "-i",
        video,
        "-frames:v",
        "1",
        "-vf",
        f"scale={w}:{h}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3).copy() if len(out) == w * h * 3 else None


def draw(img, res, t, tracks, sx):
    df = res.tracks.df
    if len(res.tracks.frame_times):
        tt = res.tracks.frame_times[np.argmin(np.abs(res.tracks.frame_times - t))]
        rows = df[np.isclose(df["t"], tt)]
        for r in rows.itertuples(index=False):
            hot = int(r.track_id) in tracks
            c = (0, 0, 255) if hot else (160, 160, 160)
            p1, p2 = (int(r.x1 * sx), int(r.y1 * sx)), (int(r.x2 * sx), int(r.y2 * sx))
            cv2.rectangle(img, p1, p2, c, 2 if hot else 1)
            if hot:
                cv2.putText(
                    img, f"#{int(r.track_id)} {r.speed:.1f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
                )
    sc = res.scene
    for cw in sc.crosswalks:
        cv2.polylines(img, [(cw * sx).astype(np.int32)], True, (0, 200, 0), 1)
    for sl in sc.stop_lines:
        cv2.polylines(img, [(sl.points * sx).astype(np.int32)], False, (0, 0, 255), 1)
    state = {sid: res.signal.state_at(sid, t) for sid in res.signal.segments}
    cv2.putText(img, f"t={t:.2f}s  signal={state}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/review")
    ap.add_argument("--candidates", type=float, default=None, help="also review candidates with at least this confidence")
    args = ap.parse_args(argv)
    res = EventPipeline.from_config().run_full(args.video)
    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    gates = {k: v.get("min_confidence", 0.5) for k, v in res_cfg(res).items()}
    items = [
        s
        for s in res.segments
        if s.confidence >= gates.get(s.label, 0.5) or (args.candidates is not None and s.confidence >= args.candidates)
    ]
    items.sort(key=lambda s: (s.label, s.start))
    w, h = WIDTH, int(round(res.meta.height * WIDTH / res.meta.width / 2)) * 2
    sx = WIDTH / res.meta.width
    index = []
    for k, s in enumerate(items):
        tracks = set(int(t) for t in s.info.get("tracks", []))
        tiles = []
        for t in (s.start, (s.start + s.end) / 2, s.end):
            f = grab(args.video, t, (w, h))
            tiles.append(draw(f, res, t, tracks, sx) if f is not None else np.zeros((h, w, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], np.zeros_like(tiles[2])])])
        passed = s.confidence >= gates.get(s.label, 0.5)
        caption = f"{s.label} {s.start:.2f}-{s.end:.2f}s conf {s.confidence:.2f} {'EVENT' if passed else 'below gate'} {json.dumps(s.info, default=str)[:120]}"
        cv2.putText(sheet, caption, (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        name = f"{k:03d}_{s.label}_{s.start:07.2f}.jpg"
        cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        index.append(
            {
                "file": name,
                "label": s.label,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "confidence": round(s.confidence, 2),
                "passed": passed,
                "info": s.info,
            }
        )
    (out / "index.json").write_text(
        json.dumps(
            {
                "events": res.events,
                "align": {k: v for k, v in res.align.items() if k != "H"},
                "signal": res.signal.to_json(),
                "items": index,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[review] {Path(args.video).name}: {len(res.events)} events, {len(index)} sheets -> {out}", file=sys.stderr)
    return 0


def res_cfg(res) -> dict:
    from roadsight.config import load_config

    return load_config().get("events", {})


if __name__ == "__main__":
    sys.exit(main())
