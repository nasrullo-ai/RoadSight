"""failure_to_yield: a moving vehicle drives into a crosswalk while a pedestrian is walking on that
crosswalk close to the vehicle's path.

Not a violation when the vehicle has a green signal governing that crosswalk (pedestrians should be
waiting), when the "pedestrian" is really a rider or a passenger (box overlapping a vehicle, or
moving faster than walking), or when the person is merely waiting at the edge.
"""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration
from roadsight.scene.geometry import points_in_polygon


@register
class FailureToYield(EventRule):
    label = "failure_to_yield"

    def detect(self, tracks, signal, meta):
        drawn = bool(self.scene.crosswalks)
        if not drawn and not self.scene.has_crosswalks():
            return []
        min_speed = self.p("min_speed", 0.5)
        near_bl = self.p("near_bl", 2.5)
        walk_min = self.p("walk_speed_min", 0.15)
        walk_max = self.p("walk_speed_max", 3.0)
        min_sec = self.p("min_sec", 0.3)
        max_len = self.p("max_len_sec", 6.0)

        veh = tracks.of_kind("vehicle", reliable=True)
        veh = veh[veh["cls"].isin(self.p("vehicle_classes", [2, 5, 7]))]  # scooters weave through crossings
        ped = tracks.of_kind("person", reliable=True)
        if veh.empty or ped.empty:
            return []
        all_veh = tracks.of_kind("vehicle")
        veh_boxes = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in all_veh.groupby("frame", sort=True)}

        px, py = ped["foot_x"].to_numpy(), ped["foot_y"].to_numpy()
        p_idx = self.cw_index(px, py, drawn)
        psp = ped["speed"].to_numpy()
        ok = (p_idx >= 0) & (psp >= walk_min) & (psp <= walk_max)
        # Drop people whose foot point lies inside a vehicle box (riders, passengers, occupants).
        frames = ped["frame"].to_numpy()
        for i in np.flatnonzero(ok):
            b = veh_boxes.get(int(frames[i]))
            if b is not None and ((px[i] >= b[:, 0]) & (px[i] <= b[:, 2]) & (py[i] >= b[:, 1]) & (py[i] <= b[:, 3])).any():
                ok[i] = False
        walkers: dict[int, list[tuple[float, float, int]]] = {}
        for i in np.flatnonzero(ok):
            walkers.setdefault(int(frames[i]), []).append((px[i], py[i], int(p_idx[i])))

        conf = 0.75 if drawn else 0.55
        out = []
        for tid, g in veh.groupby("track_id", sort=True):
            x, y = g["foot_x"].to_numpy(), g["foot_y"].to_numpy()
            idx = self.cw_index(x, y, drawn)
            if (idx < 0).all():
                continue
            t = g["t"].to_numpy()
            fr = g["frame"].to_numpy()
            sp = g["speed"].to_numpy()
            vx, vy = g["vx"].to_numpy(), g["vy"].to_numpy()
            scale = tracks.scale(y)
            for c in sorted(set(idx[idx >= 0].tolist())):
                for s, e in runs_min_duration(idx == c, t, min_sec, max_gap_sec=0.5):
                    entry = (t >= t[s]) & (t <= t[s] + 1.0)
                    if np.median(sp[entry]) < min_speed:
                        continue  # crept onto the zebra from a queue, not driving through
                    e = min(e, int(np.searchsorted(t, t[s] + max_len, side="right")) - 1)
                    sig = self.scene.crosswalk_signals[c] if drawn and c < len(self.scene.crosswalk_signals) else None
                    if sig and signal.state_at(sig, float(t[s])) == "green":
                        continue  # the vehicle has right of way
                    if self._conflict(
                        walkers, fr[s : e + 1], x[s : e + 1], y[s : e + 1], vx[s : e + 1], vy[s : e + 1], scale[s : e + 1], c, near_bl
                    ):
                        a, b = (s, e) if not drawn else self._on_crossing(g, self.scene.crosswalks[c], s, e)
                        b = min(b, int(np.searchsorted(t, t[a] + max_len, side="right")) - 1)
                        out.append(Segment(float(t[a]), float(t[b]), self.label, conf, {"tracks": [int(tid)], "crosswalk": int(c)}))
        return out

    @staticmethod
    def _on_crossing(g, polygon, s: int, e: int, frac: float = 0.35) -> tuple[int, int]:
        """Widen [s, e] to the rows where any part of the vehicle's ground footprint touches the crossing,
        so the segment runs from "vehicle enters the crossing" to "vehicle leaves the crossing"."""
        x1, x2, y2 = g["x1"].to_numpy(), g["x2"].to_numpy(), g["y2"].to_numpy()
        top = y2 - frac * (y2 - g["y1"].to_numpy())
        touch = np.zeros(len(g), dtype=bool)
        for px in (x1, (x1 + x2) / 2, x2):
            for py in (top, y2):
                touch |= points_in_polygon(px, py, polygon)
        touch[s : e + 1] = True
        while s > 0 and touch[s - 1]:
            s -= 1
        while e < len(g) - 1 and touch[e + 1]:
            e += 1
        return s, e

    def cw_index(self, x, y, drawn: bool) -> np.ndarray:
        if drawn:
            return self.scene.crosswalk_index(x, y)
        return np.where(self.scene.in_crosswalk(x, y), 0, -1)

    @staticmethod
    def _conflict(walkers, frames, x, y, vx, vy, scale, c, near_bl) -> bool:
        """A walker on the same crosswalk within ``near_bl`` and not behind the vehicle."""
        for f, xi, yi, vxi, vyi, si in zip(frames, x, y, vx, vy, scale):
            for pxi, pyi, pc in walkers.get(int(f), []):
                if pc != c:
                    continue
                dx, dy = (pxi - xi) / si, (pyi - yi) / si
                if np.hypot(dx, dy) > near_bl:
                    continue
                v = np.hypot(vxi, vyi)
                if v < 1e-6 or (dx * vxi + dy * vyi) / v > -0.5:
                    return True
        return False
