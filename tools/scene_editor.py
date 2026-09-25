"""Click-to-draw editor for configs/scene.json on a median background frame.

Pick an element with its key, left-click to add points, Enter to finish the shape,
Backspace to drop the last point, z to delete the last finished shape of the current element,
s to save, q to quit (saves).

  c carriageway   l lane (then type direction by clicking 2 more points: tail, head)
  p stop line (2+ pts)   w crosswalk   o solid line   u no-U-turn zone   x exit zone
  g signal ROI (2 pts: corners)   k queue zone   i intersection   f fixed object

Allowed movements and ids are edited in the JSON afterwards.

    python tools/scene_editor.py --video data/samples/test_001.mp4 [--scene configs/scene.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MODES = {
    "c": ("carriageway", "poly"),
    "l": ("lanes", "lane"),
    "p": ("stop_lines", "line"),
    "w": ("crosswalks", "poly"),
    "o": ("solid_lines", "polyline"),
    "u": ("no_u_turn_zones", "poly"),
    "x": ("exit_zones", "zone"),
    "g": ("signal_rois", "rect"),
    "k": ("queue_zones", "poly"),
    "i": ("intersection", "poly"),
    "f": ("fixed_objects", "poly"),
}
COLORS = {
    "carriageway": (120, 80, 30),
    "lanes": (0, 200, 0),
    "stop_lines": (0, 0, 255),
    "crosswalks": (200, 200, 200),
    "solid_lines": (0, 255, 255),
    "no_u_turn_zones": (0, 120, 255),
    "exit_zones": (255, 0, 255),
    "signal_rois": (255, 255, 0),
    "queue_zones": (255, 120, 0),
    "intersection": (100, 100, 255),
    "fixed_objects": (80, 80, 80),
}


def median_frame(path: str, n: int = 25) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scene", default="configs/scene.json")
    args = ap.parse_args(argv)
    bg = median_frame(args.video)
    h, w = bg.shape[:2]
    path = Path(args.scene)
    scene = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    scene.update({"version": 1, "normalized": True, "image_size": [w, h]})
    for key, _ in MODES.values():
        scene.setdefault(key, [])
    state = {"mode": "c", "pts": []}

    def norm(p):
        return [round(p[0] / w, 5), round(p[1] / h, 5)]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x, y))

    def finish():
        key, kind = MODES[state["mode"]]
        pts = state["pts"]
        n = len(scene[key])
        if kind == "poly" and len(pts) >= 3:
            scene[key].append([norm(p) for p in pts])
        elif kind == "polyline" and len(pts) >= 2:
            scene[key].append([norm(p) for p in pts])
        elif kind == "line" and len(pts) >= 2:
            scene[key].append(
                {
                    "id": f"S{n + 1}",
                    "points": [norm(p) for p in pts],
                    "signal": scene["signal_rois"][0]["id"] if scene["signal_rois"] else None,
                }
            )
        elif kind == "zone" and len(pts) >= 3:
            scene[key].append({"id": f"Z{n + 1}", "polygon": [norm(p) for p in pts]})
        elif kind == "rect" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            scene[key].append({"id": f"sig{n + 1}", "rect": [min(x1, x2) / w, min(y1, y2) / h, max(x1, x2) / w, max(y1, y2) / h]})
        elif kind == "lane" and len(pts) >= 5:
            poly, (tx, ty), (hx, hy) = pts[:-2], pts[-2], pts[-1]
            scene[key].append({"id": f"L{n + 1}", "polygon": [norm(p) for p in poly], "direction": [(hx - tx) / w, (hy - ty) / h]})
        else:
            print(f"not enough points for {key} ({kind})")
            return
        print(f"+ {key}")
        state["pts"] = []

    def draw() -> np.ndarray:
        img = bg.copy()
        for key, items in scene.items():
            if key not in COLORS:
                continue
            c = COLORS[key]
            for it in items:
                pts = it.get("polygon") or it.get("points") if isinstance(it, dict) else it
                if isinstance(it, dict) and "rect" in it:
                    x1, y1, x2, y2 = it["rect"]
                    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), c, 2)
                    continue
                arr = np.array([[p[0] * w, p[1] * h] for p in pts], np.int32)
                closed = key not in ("stop_lines", "solid_lines")
                cv2.polylines(img, [arr], closed, c, 2)
                if isinstance(it, dict) and "direction" in it:
                    cx, cy = arr.mean(0)
                    d = np.array(it["direction"]) * [w, h]
                    d = d / (np.linalg.norm(d) + 1e-9) * 40
                    cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + d[0]), int(cy + d[1])), c, 2)
        for p in state["pts"]:
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        if len(state["pts"]) > 1:
            cv2.polylines(img, [np.array(state["pts"], np.int32)], False, (0, 0, 255), 1)
        cv2.putText(
            img,
            f"mode: {MODES[state['mode']][0]}  (keys: {' '.join(MODES)}; Enter finish, s save, q quit)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return img

    win = "scene editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, draw())
        key = cv2.waitKey(30) & 0xFF
        ch = chr(key) if key < 128 else ""
        if ch == "q":
            break
        if ch in MODES:
            state["mode"], state["pts"] = ch, []
        elif key in (13, 10):
            finish()
        elif key == 8 and state["pts"]:
            state["pts"].pop()
        elif ch == "z":
            k = MODES[state["mode"]][0]
            if scene[k]:
                scene[k].pop()
        elif ch == "s":
            path.write_text(json.dumps(scene, indent=1), encoding="utf-8")
            print(f"saved {path}")
    path.write_text(json.dumps(scene, indent=1), encoding="utf-8")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
