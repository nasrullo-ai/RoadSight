"""red_light: a vehicle drives across its stop line while the signal is red (after a grace period).

The crossing must be a real run through the line: the front is clearly before the line
(``before_bl``), then clearly past it (``after_bl``) within a few seconds while moving. A vehicle
waiting with its bumper on the line, whose position jitters by a few pixels, never qualifies.
"""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.events.rules._stopline import approach_side, signed_distance_bl, within_line_span
from roadsight.scene.geometry import angle_diff_deg


@register
class RedLight(EventRule):
    label = "red_light"

    def detect(self, tracks, signal, meta):
        if not self.scene.stop_lines or not signal.known():
            return []
        grace = self.p("grace_sec", 0.5)
        clear_bl = self.p("clear_bl", 4.0)
        before_bl = self.p("before_bl", 0.3)
        after_bl = self.p("after_bl", 1.0)
        window = self.p("cross_window_sec", 3.0)
        min_speed = self.p("min_speed", 0.5)
        max_len = self.p("max_len_sec", 8.0)
        rtor_deg = self.p("right_turn_deg", 60.0)
        veh = tracks.of_kind("vehicle", reliable=True)
        groups = [g for _, g in veh.groupby("track_id", sort=True)]
        out = []
        for line in self.scene.stop_lines:
            side_from = approach_side(line, groups, tracks.scale)
            for g in groups:
                t = g["t"].to_numpy()
                d = signed_distance_bl(g, line, side_from, tracks.scale)
                span = within_line_span(g, line, 0.5, tracks.scale)
                sp = g["speed"].to_numpy()
                k = self._crossing(t, d, span, sp, before_bl, after_bl, window, min_speed)
                if k is None:
                    continue
                tc = float(t[k])
                if signal.state_at(line.signal, tc) != "red":
                    continue
                onset = signal.red_onset_before(line.signal, tc)
                if onset is None or tc - onset < grace:
                    continue
                if self.scene.right_turn_on_red and self._turns_right(g, k, rtor_deg):
                    continue
                past = np.flatnonzero((t > tc) & (d >= clear_bl))
                end = float(t[past[0]]) if len(past) else float(t[-1])
                out.append(Segment(tc, min(end, tc + max_len), self.label, 0.8, {"tracks": [int(g["track_id"].iloc[0])], "line": line.id}))
        return out

    @staticmethod
    def _crossing(t, d, span, sp, before_bl, after_bl, window, min_speed) -> int | None:
        """Index where the front passes the line during a clean run from before to well past it."""
        for k in np.flatnonzero((d[1:] >= 0) & (d[:-1] < 0)) + 1:
            if not span[k] or sp[k] < min_speed:
                continue
            was_before = (t >= t[k] - window) & (t < t[k]) & (d <= -before_bl)
            gets_past = (t > t[k]) & (t <= t[k] + window) & (d >= after_bl)
            if was_before.any() and gets_past.any():
                return int(k)
        return None

    @staticmethod
    def _turns_right(g, k, deg) -> bool:
        t = g["t"].to_numpy()
        h = g["heading"].to_numpy()
        sp = g["speed"].to_numpy()
        m = (t >= t[k]) & (t <= t[k] + 5.0) & (sp > 0.5)
        if m.sum() < 2:
            return False
        dh = np.cumsum(angle_diff_deg(h[m][1:], h[m][:-1]))
        return bool(dh.max(initial=0) > deg)  # clockwise on screen (y down) == right turn
