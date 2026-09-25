"""fire_smoke: flame-coloured, flickering regions persisting >= 3 s (colour + motion heuristic).

Disabled by default: without a trained fire/smoke detector the heuristic is not precise enough
for the macro-F1 gate. Enable it in ``configs/default.yaml`` once validated on real clips.
"""

from __future__ import annotations

import cv2
import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration


@register
class FireSmoke(EventRule):
    label = "fire_smoke"

    def detect(self, tracks, signal, meta):
        samples = getattr(tracks, "color_samples", None)  # list of (t, BGR thumbnail)
        if not samples or len(samples) < 4:
            return []
        min_frac = self.p("min_area_frac", 0.002)
        flicker = self.p("flicker_thr", 12.0)
        min_sec = self.p("min_sec", 3.0)
        ts = np.array([t for t, _ in samples])
        fire = np.zeros(len(ts), dtype=bool)
        prev_v = None
        for k, (_, img) in enumerate(samples):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            mask = (h <= 30) & (s >= 120) & (v >= 200)
            if prev_v is not None and mask.mean() >= min_frac:
                change = np.abs(v.astype(np.int16) - prev_v.astype(np.int16))[mask].mean()
                fire[k] = change >= flicker
            prev_v = v
        return [
            Segment(float(ts[s]), float(ts[e]), self.label, 0.5, {"source": "heuristic"})
            for s, e in runs_min_duration(fire, ts, min_sec, max_gap_sec=1.5)
        ]
