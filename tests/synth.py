"""Synthetic tracks and videos for tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roadsight.perception.tracktable import TrackTable

FPS = 12.5
W, H = 1280, 720
CAR = (80.0, 60.0)  # box w, h -> size ~69 px = 1 body length


def make_tracks(objs, duration: float, fps: float = FPS, cfg: dict | None = None) -> TrackTable:
    """``objs``: list of (track_id, cls, fn, t0, t1) where fn(t) -> (cx, foot_y, w, h)."""
    times = np.arange(0.0, duration, 1.0 / fps)
    rec = []
    for tid, cls, fn, t0, t1 in objs:
        for k, t in enumerate(times):
            if t0 <= t <= t1:
                cx, fy, w, h = fn(t)
                rec.append([k, t, tid, cls, 0.9, cx - w / 2, fy - h, cx + w / 2, fy])
    cfg = cfg or {"min_track_sec": 1.0, "max_interp_gap_sec": 1.0}
    return TrackTable.from_records(np.array(rec), times, np.arange(len(times)), W, H, cfg)


def piecewise(points):
    """Linear interpolation through (t, x, y) keyframes; returns fn(t) -> (x, y)."""
    pts = np.asarray(points, dtype=float)

    def fn(t):
        return float(np.interp(t, pts[:, 0], pts[:, 1])), float(np.interp(t, pts[:, 0], pts[:, 2]))

    return fn


def car(path, size=CAR):
    def fn(t):
        x, y = path(t)
        return x, y, size[0], size[1]

    return fn


def write_video(path: Path, seconds: float = 10.0, fps: float = 25.0, size=(640, 360), source: Path | None = None) -> Path:
    """Write a short test clip: the first ``seconds`` of ``source`` (resized) or moving shapes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source)) if source else None
    if cap is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(seconds * fps)
    for i in range(n):
        frame = None
        if cap is not None:
            ok, frame = cap.read()
            frame = cv2.resize(frame, size) if ok else None
        if frame is None:
            frame = np.full((size[1], size[0], 3), 90, np.uint8)
            x = int((i * 7) % size[0])
            cv2.rectangle(frame, (x, 150), (x + 60, 190), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    if cap is not None:
        cap.release()
    return path
