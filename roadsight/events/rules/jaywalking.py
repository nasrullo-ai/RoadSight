"""jaywalking: a pedestrian walks onto the carriageway away from every crosswalk.

The person must get clearly into the road (``min_depth_bl`` from the kerb at some point) and stay
on it for ``min_sec``; people waiting at the kerb, riders and people next to vehicles do not count.
"""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration


@register
class Jaywalking(EventRule):
    label = "jaywalking"

    def detect(self, tracks, signal, meta):
        if not self.scene.has_carriageway():
            return []
        min_sec = self.p("min_sec", 1.0)
        gap = self.p("merge_gap_sec", 2.0)
        edge_bl = self.p("edge_margin_bl", 0.3)
        cw_buffer_bl = self.p("crosswalk_buffer_bl", 0.25)
        exit_bl = self.p("vehicle_exit_bl", 2.0)
        min_depth_bl = self.p("min_depth_bl", 1.0)
        walk_max = self.p("walk_speed_max", 3.0)

        ped = tracks.of_kind("person", reliable=True)
        ped = ped[(ped["track_size"] >= self.p("min_size_frac", 0.03) * tracks.height) & ~ped["edge"]]  # far field and cut-off boxes
        if ped.empty:
            return []
        veh = tracks.of_kind("vehicle")
        still = veh[veh["speed"] < 0.2]
        still_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in still.groupby("frame", sort=True)}
        all_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in veh.groupby("frame", sort=True)}

        x = ped["foot_x"].to_numpy()
        y = ped["foot_y"].to_numpy()
        scale = tracks.scale(y)
        depth = self.scene.depth_in_carriageway(x, y) / scale
        on_road = self.scene.in_carriageway(x, y) & (depth >= edge_bl) & (ped["speed"].to_numpy() <= walk_max)
        if self.scene.has_crosswalks():
            on_road &= ~self.scene.in_crosswalk(x, y, buffer_px=float(np.median(scale)) * cw_buffer_bl)

        # Drivers/passengers next to a stopped vehicle do not count until they walk away.
        frames = ped["frame"].to_numpy()
        near_car = np.zeros(len(ped), dtype=bool)
        for i in np.flatnonzero(on_road):
            boxes = still_by_frame.get(int(frames[i]))
            anyb = all_by_frame.get(int(frames[i]))
            if anyb is not None and ((x[i] >= anyb[:, 0]) & (x[i] <= anyb[:, 2]) & (y[i] >= anyb[:, 1]) & (y[i] <= anyb[:, 3])).any():
                near_car[i] = True  # foot inside a vehicle box: rider or passenger
                continue
            if boxes is None:
                continue
            pad = exit_bl * scale[i]
            near_car[i] = bool(
                (
                    (x[i] >= boxes[:, 0] - pad) & (x[i] <= boxes[:, 2] + pad) & (y[i] >= boxes[:, 1] - pad) & (y[i] <= boxes[:, 3] + pad)
                ).any()
            )
        on_road &= ~near_car

        conf = 0.8 if (self.scene.carriageway and self.scene.crosswalks) else 0.6
        out = []
        tids = ped["track_id"].to_numpy()
        t_all = ped["t"].to_numpy()
        for tid in np.unique(tids):
            m = tids == tid
            t = t_all[m]
            dm = depth[m]
            for s, e in runs_min_duration(on_road[m], t, min_sec, max_gap_sec=gap):
                if dm[s : e + 1].max() < min_depth_bl:
                    continue  # stayed at the kerb
                out.append(Segment(float(t[s]), float(t[e]), self.label, conf, {"tracks": [int(tid)]}))
        return out
