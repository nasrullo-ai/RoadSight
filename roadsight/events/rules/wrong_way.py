"""wrong_way: a vehicle moving against its lane's flow for >= 1.5 s."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration


@register
class WrongWay(EventRule):
    label = "wrong_way"

    def detect(self, tracks, signal, meta):
        min_speed = self.p("min_speed", 0.5)
        cos_thr = self.p("cos_thr", -0.5)
        min_sec = self.p("min_sec", 1.5)
        min_strength = self.p("min_strength", 0.6)
        min_disp_bl = self.p("min_displacement_bl", 2.0)
        min_med_speed = self.p("min_median_speed", 0.8)
        min_real = self.p("min_real_frac", 0.7)
        reversal_sec = self.p("reversal_window_sec", 3.0)

        veh = tracks.of_kind("vehicle", reliable=True)
        out = []
        for tid, g in veh.groupby("track_id", sort=True):
            sp = g["speed"].to_numpy()
            if (sp > min_speed).sum() < 3:
                continue
            x = g["sx"].to_numpy()
            y = g["sy"].to_numpy()
            dirs, strength = self.scene.lane_direction(x, y, exclude_track=int(tid))
            u = np.stack([g["vx"].to_numpy(), g["vy"].to_numpy()], 1) / np.maximum(sp, 1e-9)[:, None]
            cos = (u * dirs).sum(1)
            against = (sp > min_speed) & (strength >= min_strength) & (cos < cos_thr)
            t = g["t"].to_numpy()
            real = ~g["interp"].to_numpy()
            along = (sp > min_speed) & (strength >= min_strength) & (cos > 0.5)
            for s, e in runs_min_duration(against, t, min_sec, max_gap_sec=0.5):
                disp = np.hypot(x[e] - x[s], y[e] - y[s]) / float(tracks.scale((y[s] + y[e]) / 2))
                if disp < min_disp_bl or np.median(sp[s : e + 1]) < min_med_speed or real[s : e + 1].mean() < min_real:
                    continue
                # Moving with the flow just before and then against it is an ID switch, not a wrong-way driver.
                before = (t >= t[s] - reversal_sec) & (t < t[s]) & along
                if before.sum() * (np.median(np.diff(t)) if len(t) > 1 else 0.1) >= 1.0:
                    continue
                conf = float(np.clip(np.mean(strength[s : e + 1]) * np.mean(-cos[s : e + 1]), 0, 1))
                out.append(Segment(float(t[s]), float(t[e]), self.label, conf, {"tracks": [int(tid)]}))
        return out
