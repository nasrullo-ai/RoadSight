"""solid_line_crossing: a vehicle's wheel line crosses a solid lane line and it ends up on the other side."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs
from roadsight.scene.geometry import distance_to_polyline, footprint, side_of_polyline


def line_through_boxes(boxes: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Whether ``line`` passes through each box: its corners lie on both sides and the box is near it."""
    corners = [(boxes[:, 0], boxes[:, 1]), (boxes[:, 2], boxes[:, 1]), (boxes[:, 0], boxes[:, 3]), (boxes[:, 2], boxes[:, 3])]
    sides = np.stack([side_of_polyline(cx, cy, line) for cx, cy in corners], 1)
    mixed = sides.min(1) != sides.max(1)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return mixed & (distance_to_polyline(cx, cy, line) <= diag / 2)


@register
class SolidLineCrossing(EventRule):
    label = "solid_line_crossing"

    def detect(self, tracks, signal, meta):
        if not self.scene.solid_lines:
            return []
        hold = self.p("hold_sec", 1.0)
        min_speed = self.p("min_speed", 0.3)
        frac = self.p("footprint_frac", 0.35)
        veh = tracks.of_kind("vehicle", reliable=True)
        out = []
        for tid, g in veh.groupby("track_id", sort=True):
            if g["speed"].max() < min_speed:
                continue
            t = g["t"].to_numpy()
            x, y = g["sx"].to_numpy(), g["sy"].to_numpy()
            fp = footprint(g[["x1", "y1", "x2", "y2"]].to_numpy(), frac)
            for line in self.scene.solid_lines:
                touching = line_through_boxes(fp, line)
                if not touching.any():
                    continue
                side = side_of_polyline(x, y, line)
                for s, e in runs(touching):
                    before = (t < t[s]) & (t >= t[s] - hold - 5.0)
                    after = (t > t[e]) & (t <= t[e] + hold + 5.0)
                    if not before.any() or not after.any():
                        continue
                    sb = np.sign(side[before].mean())
                    sa = np.sign(side[after].mean())
                    stable_after = (t[after][-1] - t[after][0]) >= hold and np.all(side[after] == sa)
                    if sb != 0 and sa != 0 and sb != sa and stable_after:
                        out.append(Segment(float(t[s]), float(t[e]), self.label, 0.7, {"tracks": [int(tid)]}))
        return out
