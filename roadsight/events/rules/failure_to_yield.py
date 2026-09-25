"""failure_to_yield: a vehicle enters a crosswalk while a pedestrian is on it or about to step on it."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration


@register
class FailureToYield(EventRule):
    label = "failure_to_yield"

    def detect(self, tracks, signal, meta):
        if not self.scene.has_crosswalks():
            return []
        min_speed = self.p("min_speed", 0.3)
        edge_bl = self.p("edge_buffer_bl", 0.25)
        near_bl = self.p("near_bl", 4.0)
        min_sec = self.p("min_sec", 0.3)

        veh = tracks.of_kind("vehicle", reliable=True)
        ped = tracks.of_kind("person", reliable=True)
        if veh.empty or ped.empty:
            return []
        px, py = ped["foot_x"].to_numpy(), ped["foot_y"].to_numpy()
        pscale = tracks.scale(py)
        p_in = self.scene.in_crosswalk(px, py, buffer_px=float(np.median(pscale)) * edge_bl)
        ped_on = ped[p_in]
        ped_by_frame = {int(f): g[["foot_x", "foot_y"]].to_numpy() for f, g in ped_on.groupby("frame", sort=True)}
        conf = 0.7 if self.scene.crosswalks else 0.55

        out = []
        for tid, g in veh.groupby("track_id", sort=True):
            x, y = g["foot_x"].to_numpy(), g["foot_y"].to_numpy()
            in_cw = self.scene.in_crosswalk(x, y)
            if not in_cw.any():
                continue
            t = g["t"].to_numpy()
            frames = g["frame"].to_numpy()
            sp = g["speed"].to_numpy()
            scale = tracks.scale(y)
            for s, e in runs_min_duration(in_cw, t, min_sec, max_gap_sec=0.5):
                if sp[s : e + 1].max() < min_speed:
                    continue
                conflict = False
                for i in range(s, e + 1):
                    peds = ped_by_frame.get(int(frames[i]))
                    if peds is not None and (np.hypot(peds[:, 0] - x[i], peds[:, 1] - y[i]) < near_bl * scale[i]).any():
                        conflict = True
                        break
                if conflict:
                    out.append(Segment(float(t[s]), float(t[e]), self.label, conf, {"tracks": [int(tid)]}))
        return out
