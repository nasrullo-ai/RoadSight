"""stop_line: an approaching vehicle stops clearly past its stop line (but not inside the intersection) during red."""

from __future__ import annotations

from roadsight.events.base import EventRule, Segment, register, runs_min_duration
from roadsight.events.rules._stopline import approach_side, signed_distance_bl, within_line_span
from roadsight.scene.geometry import points_in_any


@register
class StopLineViolation(EventRule):
    label = "stop_line"

    def detect(self, tracks, signal, meta):
        if not self.scene.stop_lines or not signal.known():
            return []
        v_stop = self.p("stop_speed", 0.1)
        min_stop = self.p("min_stop_sec", 2.0)
        min_past = self.p("min_past_bl", 0.4)
        max_past = self.p("max_past_bl", 3.0)
        veh = tracks.of_kind("vehicle", reliable=True)
        groups = [g for _, g in veh.groupby("track_id", sort=True)]
        out = []
        for line in self.scene.stop_lines:
            side_from = approach_side(line, groups, tracks.scale)
            for g in groups:
                t = g["t"].to_numpy()
                d = signed_distance_bl(g, line, side_from, tracks.scale)
                if not (d < -0.3).any():
                    continue  # never seen approaching this line: cross traffic, not a violation
                bad = (d >= min_past) & (d <= max_past) & within_line_span(g, line, 0.3, tracks.scale) & (g["speed"].to_numpy() < v_stop)
                if self.scene.intersection:
                    bad &= ~points_in_any(g["sx"].to_numpy(), g["sy"].to_numpy(), self.scene.intersection)
                for s, e in runs_min_duration(bad, t, min_stop):
                    red = [tt for tt in t[s : e + 1] if signal.state_at(line.signal, tt) == "red"]
                    if len(red) < 0.5 * (e - s + 1):
                        continue
                    green = signal.next_change(line.signal, float(t[s]), "green")
                    end = float(t[e]) if green is None else min(float(t[e]), green)
                    if end > t[s]:
                        out.append(Segment(float(t[s]), end, self.label, 0.7, {"tracks": [int(g["track_id"].iloc[0])], "line": line.id}))
        return out
