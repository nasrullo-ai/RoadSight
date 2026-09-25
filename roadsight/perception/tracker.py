"""ByteTrack wrapper around the Ultralytics implementation.

Track IDs are remapped to a per-instance counter, so IDs do not depend on Ultralytics' global
counter (which is shared by every tracker in the process). Returned boxes are the matched
detection boxes, not Kalman states.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _Dets:
    """Minimal stand-in for Ultralytics ``Boxes`` as used by ``BYTETracker.update``."""

    def __init__(self, dets: np.ndarray) -> None:
        self.data = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        x1, y1, x2, y2 = self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3]
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        self.xyxy = self.data[:, :4]
        self.conf = self.data[:, 4]
        self.cls = self.data[:, 5]

    def __len__(self) -> int:
        return len(self.data)


class Tracker:
    def __init__(self, cfg: dict) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=float(cfg.get("track_high_thresh", 0.5)),
            track_low_thresh=float(cfg.get("track_low_thresh", 0.1)),
            new_track_thresh=float(cfg.get("new_track_thresh", 0.6)),
            track_buffer=int(cfg.get("track_buffer", 30)),
            match_thresh=float(cfg.get("match_thresh", 0.8)),
            fuse_score=bool(cfg.get("fuse_score", True)),
        )
        # frame_rate=30 makes max_time_lost == track_buffer processed frames.
        self._bt = BYTETracker(args, frame_rate=30)
        self._ids: dict[int, int] = {}

    def update(self, dets: np.ndarray) -> np.ndarray:
        """dets (N, 6) x1,y1,x2,y2,conf,cls -> tracks (M, 7) x1,y1,x2,y2,track_id,conf,cls."""
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        res = self._bt.update(_Dets(dets))
        if res is None or len(res) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        res = np.asarray(res, dtype=np.float64)
        out = np.zeros((len(res), 7), dtype=np.float64)
        for k, row in enumerate(res):
            det_idx = int(row[7])
            ext = int(row[4])
            if ext not in self._ids:
                self._ids[ext] = len(self._ids) + 1
            src = dets[det_idx] if 0 <= det_idx < len(dets) else np.concatenate([row[:4], row[5:7]])
            out[k, :4] = src[:4]
            out[k, 4] = self._ids[ext]
            out[k, 5] = src[4]
            out[k, 6] = src[5]
        return out
