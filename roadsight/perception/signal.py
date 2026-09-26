"""Traffic-light state from the colour of a signal-head ROI (SPEC 7.6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

RED, AMBER, GREEN, UNKNOWN = "red", "amber", "green", "unknown"
_STATES = (RED, AMBER, GREEN)


def classify_roi(frame: np.ndarray, rect: tuple[int, int, int, int], min_frac: float = 0.0015) -> str:
    """Colour of the lit lamp inside ``rect`` (x1, y1, x2, y2).

    A lamp counts as lit when it is clearly brighter than the rest of the signal head (adaptive to noon
    sun and dusk) and saturated. Green includes the teal of LED signals. ``unknown`` when no colour has
    enough lit pixels or two colours are about equally strong.
    """
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return UNKNOWN
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2].astype(np.int16)
    lit = (vch >= max(60, int(np.median(vch)) + 25)) & (sch >= 60)
    masks = {
        RED: lit & ((hch <= 12) | (hch >= 165)),
        AMBER: lit & (hch > 12) & (hch <= 32),
        GREEN: lit & (hch >= 45) & (hch <= 100),
    }
    counts = {k: int(m.sum()) for k, m in masks.items()}
    ranked = sorted(counts, key=counts.get, reverse=True)
    best, second = ranked[0], ranked[1]
    if counts[best] < max(12, min_frac * hch.size) or counts[best] < 2 * counts[second]:
        return UNKNOWN
    return best


def mode_filter(states: list[str], k: int) -> list[str]:
    """Sliding-window majority (window ``k`` samples, centred) to remove flicker."""
    if k <= 1 or not states:
        return list(states)
    half = k // 2
    out = []
    for i in range(len(states)):
        win = states[max(0, i - half) : i + half + 1]
        counts = {s: win.count(s) for s in set(win)}
        out.append(max(sorted(counts), key=lambda s: counts[s]))
    return out


@dataclass
class SignalTimeline:
    """Per signal id: list of (t_start, t_end, state) segments."""

    segments: dict[str, list[tuple[float, float, str]]] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, samples: dict[str, list[tuple[float, str]]], smooth_sec: float = 1.0) -> SignalTimeline:
        segs: dict[str, list[tuple[float, float, str]]] = {}
        for sid, seq in samples.items():
            if not seq:
                continue
            ts = [t for t, _ in seq]
            dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
            states = mode_filter([s for _, s in seq], max(1, int(round(smooth_sec / max(dt, 1e-3))) | 1))
            out: list[tuple[float, float, str]] = []
            start = ts[0]
            for i in range(1, len(ts) + 1):
                if i == len(ts) or states[i] != states[i - 1]:
                    end = ts[i] if i < len(ts) else ts[-1] + dt
                    out.append((start, end, states[i - 1]))
                    if i < len(ts):
                        start = ts[i]
            segs[sid] = out
        return cls(segs)

    def known(self, signal_id: str | None = None) -> bool:
        if signal_id is None:
            return any(any(s != UNKNOWN for _, _, s in v) for v in self.segments.values())
        return any(s != UNKNOWN for _, _, s in self.segments.get(signal_id, []))

    def state_at(self, signal_id: str | None, t: float) -> str:
        if signal_id is None:
            if len(self.segments) != 1:
                return UNKNOWN
            signal_id = next(iter(self.segments))
        for a, b, s in self.segments.get(signal_id, []):
            if a <= t < b:
                return s
        return UNKNOWN

    def red_onset_before(self, signal_id: str | None, t: float) -> float | None:
        """Start time of the red phase containing ``t`` (None if not red)."""
        if signal_id is None and len(self.segments) == 1:
            signal_id = next(iter(self.segments))
        for a, b, s in self.segments.get(signal_id or "", []):
            if a <= t < b:
                return a if s == RED else None
        return None

    def next_change(self, signal_id: str | None, t: float, to_state: str) -> float | None:
        if signal_id is None and len(self.segments) == 1:
            signal_id = next(iter(self.segments))
        for a, _, s in self.segments.get(signal_id or "", []):
            if a > t and s == to_state:
                return a
        return None

    def to_json(self) -> dict:
        return {k: [[round(a, 2), round(b, 2), s] for a, b, s in v] for k, v in self.segments.items()}
