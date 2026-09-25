"""red_light: a vehicle front crosses a stop line while its signal is red (after a grace period)."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.events.rules._stopline import approach_side, crossing_index
from roadsight.scene.geometry import angle_diff_deg, side_of_polyline


@register
class RedLight(EventRule):
    label = "red_light"

    def detect(self, tracks, signal, meta):
        if not self.scene.stop_lines or not signal.known():
            return []
        grace = self.p("grace_sec", 0.5)
        clear_bl = self.p("clear_bl", 4.0)
        rtor_deg = self.p("right_turn_deg", 60.0)
        veh = tracks.of_kind("vehicle", reliable=True)
        groups = [g for _, g in veh.groupby("track_id", sort=True)]
        out = []
        for line in self.scene.stop_lines:
            side_from = approach_side(line, groups, tracks.scale)
            for g in groups:
                k = crossing_index(g, line, side_from, tracks.scale)
                if k is None:
                    continue
                t = g["t"].to_numpy()
                tc = float(t[k])
                if signal.state_at(line.signal, tc) != "red":
                    continue
                onset = signal.red_onset_before(line.signal, tc)
                if onset is None or tc - onset < grace:
                    continue
                if self.scene.right_turn_on_red and self._turns_right(g, k, rtor_deg):
                    continue
                end = self._end(g, k, line, side_from, clear_bl, tracks.scale)
                out.append(Segment(tc, end, self.label, 0.8, {"tracks": [int(g["track_id"].iloc[0])], "line": line.id}))
        return out

    @staticmethod
    def _turns_right(g, k, deg) -> bool:
        t = g["t"].to_numpy()
        h = g["heading"].to_numpy()
        sp = g["speed"].to_numpy()
        m = (t >= t[k]) & (t <= t[k] + 5.0) & (sp > 0.5)
        if m.sum() < 2:
            return False
        d = np.cumsum(angle_diff_deg(np.diff(h[m]), 0))
        return bool(d.max(initial=0) > deg)  # clockwise on screen (y down) == right turn

    def _end(self, g, k, line, side_from, clear_bl, scale) -> float:
        t = g["t"].to_numpy()
        x = g["sx"].to_numpy()
        y = g["sy"].to_numpy()
        for i in range(k, len(t)):
            if self.scene.exit_zones and self.scene.zone_of(x[i], y[i], self.scene.exit_zones):
                return float(t[i])
            d = np.min(np.hypot(line.points[:, 0] - x[i], line.points[:, 1] - y[i]))
            past = side_of_polyline(x[i : i + 1], y[i : i + 1], line.points)[0] != side_from
            if past and d > clear_bl * float(scale(y[i])):
                return float(t[i])
        return float(t[-1])
