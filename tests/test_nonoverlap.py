"""Property test: post-processing never emits same-class overlaps or invalid bounds."""

import numpy as np

from roadsight import CLASSES
from roadsight.events.base import Segment
from roadsight.events.postprocess import postprocess


def test_random_segments_are_cleaned():
    rng = np.random.default_rng(0)
    cfg = {
        c: {
            "merge_gap": float(rng.uniform(0, 3)),
            "min_len": float(rng.uniform(0, 2)),
            "start_offset": float(rng.uniform(-1, 1)),
            "end_offset": float(rng.uniform(-1, 1)),
            "min_confidence": 0.3,
        }
        for c in CLASSES
    }
    for _ in range(500):
        duration = float(rng.uniform(1, 120))
        segs = []
        for _ in range(int(rng.integers(0, 40))):
            s = float(rng.uniform(-5, duration + 5))
            e = s + float(rng.uniform(-1, 30))
            segs.append(Segment(s, e, str(rng.choice(CLASSES)), float(rng.uniform(0, 1))))
        out = postprocess(segs, duration, cfg)
        last = {}
        for s, e, lab in out:
            assert isinstance(s, float) and isinstance(e, float)
            assert 0.0 <= s < e <= duration
            assert round(s, 2) == s and round(e, 2) == e
            assert lab in CLASSES
            assert s >= last.get(lab, -1.0)
            last[lab] = e
        assert out == sorted(out, key=lambda x: (x[0], x[1], x[2]))


def test_touching_segments_merge():
    out = postprocess([Segment(1, 3, "jaywalking"), Segment(3, 5, "jaywalking")], 10, {"jaywalking": {"merge_gap": 0, "min_len": 0}})
    assert out == [[1.0, 5.0, "jaywalking"]]


def test_low_confidence_dropped():
    out = postprocess([Segment(1, 3, "accident", 0.2)], 10, {"accident": {"min_confidence": 0.6}})
    assert out == []
