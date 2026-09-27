"""Pick one example per detected class for the website's Results page and grab a still from the annotated render.

Prefers events we verified by eye (data/review_samples.json, verdict "correct"); otherwise the longest event
of that class. Writes web/static/media/examples.json and one JPEG per class next to the renders.

    python tools/examples.py --media web/static/media
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def grab(video: Path, t: float, out: Path) -> bool:
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(video)]
    cmd += ["-frames:v", "1", "-q:v", "3", str(out)]
    return subprocess.run(cmd, check=False).returncode == 0 and out.exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default="web/static/media")
    ap.add_argument("--review", default="data/review_samples.json")
    args = ap.parse_args(argv)
    media = Path(args.media)
    index = json.loads((media / "index.json").read_text(encoding="utf-8"))
    review = json.loads((ROOT / args.review).read_text(encoding="utf-8")).get("reviewed", []) if (ROOT / args.review).exists() else []
    verified = [(r["video"], r["label"], r["t"]) for r in review if r["verdict"] == "correct"]

    candidates: dict[str, list[dict]] = {}
    for row in index:
        bundle = json.loads((media / f"{row['stem']}.json").read_text(encoding="utf-8"))
        for s, e, label in bundle["events"]:
            ok = any(v == row["video"] and lab == label and s - 1.0 <= t <= e + 1.0 for v, lab, t in verified)
            candidates.setdefault(label, []).append({"video": row["video"], "stem": row["stem"], "start": s, "end": e, "verified": ok})

    examples = []
    for label in sorted(candidates):
        best = max(candidates[label], key=lambda c: (c["verified"], min(c["end"] - c["start"], 10.0)))
        t = best["start"] + min(1.5, (best["end"] - best["start"]) / 2)
        img = f"example_{label}.jpg"
        if grab(media / f"{best['stem']}.mp4", t, media / img):
            examples.append({**best, "label": label, "image": img, "count": len(candidates[label])})
            print(
                f"[examples] {label}: {best['video']} {best['start']:.1f}-{best['end']:.1f}s verified={best['verified']}", file=sys.stderr
            )
    (media / "examples.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
