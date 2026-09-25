"""illegal_u_turn and illegal_turn: movement classification from heading change and entry/exit zones."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.scene.geometry import angle_diff_deg, points_in_any


def turn_profile(g, min_speed: float):
    """Times and cumulative heading change (deg, unwrapped) over the moving samples of a track."""
    m = g["speed"].to_numpy() > min_speed
    t = g["t"].to_numpy()[m]
    h = g["heading"].to_numpy()[m]
    if len(t) < 3:
        return t, np.zeros(len(t)), m
    cum = np.concatenate([[0.0], np.cumsum(angle_diff_deg(h[1:], h[:-1]))])
    return t, cum, m


def turn_window(t, cum, start_deg: float, settle_rate: float):
    """(start, end) of the turn: start when |change| first exceeds ``start_deg``; end when the
    heading rate stays below ``settle_rate`` deg/s for 1 s after the largest change."""
    base = np.median(cum[: max(1, min(len(cum), 5))])
    dev = np.abs(cum - base)
    if not (dev > start_deg).any():
        return None
    i0 = int(np.argmax(dev > start_deg))
    peak = int(np.argmax(dev))
    rate = np.abs(np.gradient(cum, t)) if len(t) > 2 else np.zeros(len(t))
    end = t[-1]
    calm_from = None
    for k in range(peak, len(t)):
        if rate[k] < settle_rate:
            calm_from = t[k] if calm_from is None else calm_from
            if t[k] - calm_from >= 1.0:
                end = calm_from
                break
        else:
            calm_from = None
    return float(t[i0]), float(max(end, t[peak]))


@register
class IllegalUTurn(EventRule):
    label = "illegal_u_turn"

    def detect(self, tracks, signal, meta):
        everywhere = self.p("illegal_everywhere", False)
        if not self.scene.no_u_turn_zones and not everywhere:
            return []
        min_speed = self.p("min_speed", 0.3)
        u_deg = self.p("u_turn_deg", 150.0)
        out = []
        for tid, g in tracks.of_kind("vehicle", reliable=True).groupby("track_id", sort=True):
            t, cum, m = turn_profile(g, min_speed)
            if len(t) < 3 or np.ptp(cum) < u_deg:
                continue
            win = turn_window(t, cum, self.p("start_deg", 15.0), self.p("settle_rate", 10.0))
            if win is None:
                continue
            mid = int(np.argmin(np.abs(np.abs(cum - cum[0]) - np.ptp(cum) / 2)))
            x = g["sx"].to_numpy()[m][mid : mid + 1]
            y = g["sy"].to_numpy()[m][mid : mid + 1]
            if self.scene.no_u_turn_zones and not points_in_any(x, y, self.scene.no_u_turn_zones)[0]:
                continue
            out.append(Segment(win[0], win[1], self.label, 0.75 if self.scene.no_u_turn_zones else 0.55, {"tracks": [int(tid)]}))
        return out


@register
class IllegalTurn(EventRule):
    label = "illegal_turn"

    def detect(self, tracks, signal, meta):
        if not self.scene.exit_zones or not self.scene.allowed_movements:
            return []
        min_speed = self.p("min_speed", 0.3)
        u_deg = self.p("u_turn_deg", 150.0)
        zones = self.scene.exit_zones
        out = []
        for tid, g in tracks.of_kind("vehicle", reliable=True).groupby("track_id", sort=True):
            x, y = g["sx"].to_numpy(), g["sy"].to_numpy()
            entry = next((z for z in (self.scene.zone_of(x[i], y[i], zones) for i in range(min(len(x), 10))) if z), None)
            exit_ = next(
                (z for z in (self.scene.zone_of(x[i], y[i], zones) for i in range(len(x) - 1, max(-1, len(x) - 11), -1)) if z), None
            )
            if entry is None or exit_ is None or entry == exit_ or (entry, exit_) in self.scene.allowed_movements:
                continue
            t, cum, _ = turn_profile(g, min_speed)
            if len(t) < 3 or np.ptp(cum) >= u_deg:
                continue  # U-turns belong to the other rule
            win = turn_window(t, cum, self.p("start_deg", 15.0), self.p("settle_rate", 10.0))
            if win is None:
                continue
            out.append(Segment(win[0], win[1], self.label, 0.75, {"tracks": [int(tid)], "movement": [entry, exit_]}))
        return out
