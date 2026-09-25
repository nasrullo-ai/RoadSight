"""Label events into data/dev_labels.json (ground-truth format: duration, fps, events).

Keys:  space play/pause   , / .  -1 / +1 frame   j / l  -1 / +1 s   J / L  -5 / +5 s
       [ mark start       ] mark end (then pick the class)   1-9, a-e  class for the pending segment
       u  undo last event  s save   q quit (saves)

    python tools/annotate.py --video data/samples/test_001.mp4 [--labels data/dev_labels.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadsight import CLASSES  # noqa: E402

KEYS = "123456789abcde"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"videos": {}}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for v in data["videos"].values():
        v["events"].sort(key=lambda e: (e[0], e[2]))
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"saved {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", default="data/dev_labels.json")
    args = ap.parse_args(argv)
    labels_path = Path(args.labels)
    data = load(labels_path)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = Path(args.video).name
    entry = data["videos"].setdefault(name, {"duration": round(n / fps, 3), "fps": fps, "events": []})
    idx, playing, start, pending = 0, False, None, None
    win = "annotate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, n - 1)
            playing = False
            continue
        t = idx / fps
        view = frame.copy()
        y = 24
        cv2.putText(
            view,
            f"{name}  t={t:7.2f}s  frame {idx}/{n}  {'PLAY' if playing else 'PAUSE'}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        if start is not None:
            y += 24
            cv2.putText(
                view,
                f"start marked at {start:.2f}s" + (f", end {pending:.2f}s -> pick class" if pending else ""),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
        for s, e, lab in entry["events"]:
            if s <= t <= e:
                y += 24
                cv2.putText(view, f"{lab} [{s:.2f}, {e:.2f}]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if pending is not None:
            for i, c in enumerate(CLASSES):
                cv2.putText(view, f"{KEYS[i]}: {c}", (view.shape[1] - 260, 24 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        ch = chr(key) if key < 128 else ""
        if ch == "q":
            break
        if ch == " ":
            playing = not playing
        elif ch == ",":
            idx = max(0, idx - 1)
        elif ch == ".":
            idx = min(n - 1, idx + 1)
        elif ch == "j":
            idx = max(0, idx - int(fps))
        elif ch == "l":
            idx = min(n - 1, idx + int(fps))
        elif ch == "J":
            idx = max(0, idx - int(5 * fps))
        elif ch == "L":
            idx = min(n - 1, idx + int(5 * fps))
        elif ch == "[":
            start, pending = round(t, 2), None
        elif ch == "]" and start is not None and t > start:
            pending = round(t, 2)
        elif pending is not None and ch in KEYS:
            entry["events"].append([start, pending, CLASSES[KEYS.index(ch)]])
            print(f"+ {entry['events'][-1]}")
            start, pending = None, None
        elif ch == "u" and entry["events"]:
            print(f"- {entry['events'].pop()}")
        elif ch == "s":
            save(labels_path, data)
        if playing:
            idx = min(n - 1, idx + 1)
    save(labels_path, data)
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
