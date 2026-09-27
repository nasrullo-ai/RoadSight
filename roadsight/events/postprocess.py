"""Segment post-processing (SPEC section 9).

Order: confidence cut, per-class gap merge, minimum duration, boundary offsets, clip,
same-class non-overlap, rounding. Confidence is cut first so weak candidates never stretch
confident ones.
"""

from __future__ import annotations

import math

from roadsight.events.base import Segment

DEFAULTS = {"merge_gap": 2.0, "min_len": 1.0, "start_offset": 0.0, "end_offset": 0.0, "min_confidence": 0.5}


def class_params(cfg_events: dict, label: str) -> dict:
    params = dict(DEFAULTS)
    ev = cfg_events.get(label, {}) or {}
    for k in DEFAULTS:
        if k in ev:
            params[k] = float(ev[k])
    post = ev.get("post", {}) or {}
    for k in DEFAULTS:
        if k in post:
            params[k] = float(post[k])
    return params


def suppress_near_miss(segments: list[Segment], cfg_events: dict, window: float = 3.0) -> list[Segment]:
    """Drop near misses whose road users are in a confirmed accident at about the same time (a crash is not a near miss)."""
    gate = class_params(cfg_events, "accident")["min_confidence"]
    crashes = [s for s in segments if s.label == "accident" and s.confidence >= gate]
    if not crashes:
        return segments

    def involved(s: Segment) -> set:
        return {int(t) for t in s.info.get("tracks", [])}

    return [
        s
        for s in segments
        if s.label != "near_miss"
        or not any(involved(s) & involved(a) and s.start <= a.end + window and a.start <= s.end + window for a in crashes)
    ]


def merge_same_class(segs: list[Segment], gap: float) -> list[Segment]:
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out: list[Segment] = []
    for s in segs:
        if out and s.start - out[-1].end <= gap:
            last = out[-1]
            last.end = max(last.end, s.end)
            last.confidence = max(last.confidence, s.confidence)
        else:
            out.append(Segment(s.start, s.end, s.label, s.confidence, dict(s.info)))
    return out


def postprocess(segments: list[Segment], duration: float, cfg_events: dict) -> list[list]:
    """Return the final ``[[start, end, label], ...]`` list, sorted by start time."""
    by_class: dict[str, list[Segment]] = {}
    for s in segments:
        by_class.setdefault(s.label, []).append(s)

    end_cap = math.floor(duration * 100.0) / 100.0
    final: list[Segment] = []
    for label in sorted(by_class):
        prm = class_params(cfg_events, label)
        segs = [s for s in by_class[label] if s.confidence >= prm["min_confidence"]]
        segs = merge_same_class(segs, prm["merge_gap"])
        segs = [s for s in segs if s.end - s.start >= prm["min_len"]]
        shifted = []
        for s in segs:
            start = max(0.0, s.start + prm["start_offset"])
            end = min(duration, s.end + prm["end_offset"])
            if end - start >= 0.1:
                shifted.append(Segment(start, end, label, s.confidence, s.info))
        segs = merge_same_class(shifted, 0.0)
        for s in segs:
            start = round(float(s.start), 2)
            end = min(round(float(s.end), 2), end_cap)
            if end > start:
                final.append(Segment(start, end, label, s.confidence, s.info))

    final.sort(key=lambda s: (s.start, s.end, s.label))
    assert_valid(final, duration)
    return [[float(s.start), float(s.end), str(s.label)] for s in final]


def assert_valid(segs: list[Segment], duration: float) -> None:
    last_end: dict[str, float] = {}
    for s in sorted(segs, key=lambda s: (s.label, s.start)):
        assert 0.0 <= s.start < s.end <= duration + 1e-9, (s, duration)
        assert s.start >= last_end.get(s.label, -1.0), f"same-class overlap: {s}"
        last_end[s.label] = s.end
