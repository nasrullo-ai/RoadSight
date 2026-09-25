"""Part B logic on synthetic tracks (no detector): a hard-braking conflict raises the alarm, calm traffic does not."""

import numpy as np
import pytest

from roadsight.risk import CausalRiskModel

W, H, FPS = 1280, 720, 25.0


@pytest.fixture(scope="module")
def model():
    m = CausalRiskModel.from_config()
    return m


def run(model, objs, seconds=12.0):
    """objs: list of (track_id, cls, fn(t) -> (cx, foot_y)) with 80x60 boxes."""
    model.reset({"video_id": "synthetic", "fps": FPS, "width": W, "height": H, "n_frames": int(seconds * FPS)})
    scores = []
    for k in range(int(seconds * FPS)):
        if k % model.stride:
            continue
        t = k / FPS
        rows = []
        for tid, cls, fn in objs:
            pos = fn(t)
            if pos is None:
                continue
            cx, fy = pos
            rows.append([cx - 40, fy - 60, cx + 40, fy, tid, 0.9, cls])
        scores.append(model.process_tracks(np.array(rows, dtype=np.float64).reshape(-1, 7), t))
    return np.array(scores)


def lerp(keys):
    k = np.asarray(keys, dtype=float)
    return lambda t: (float(np.interp(t, k[:, 0], k[:, 1])), float(np.interp(t, k[:, 0], k[:, 2])))


def test_hard_braking_conflict_raises_alarm(model):
    # A drives at ~3 BL/s straight at a slow car B and brakes hard only at the last moment.
    a = lerp([(0, 40, 420), (5.0, 900, 420), (5.4, 950, 420), (12, 950, 420)])
    b = lerp([(0, 1040, 420), (12, 1100, 420)])
    s = run(model, [(1, 2, a), (2, 2, b)])
    assert s.max() >= 0.5


def test_calm_parallel_traffic_stays_low(model):
    a = lerp([(0, 40, 380), (12, 1240, 380)])
    b = lerp([(0, 40, 480), (12, 1240, 480)])
    c = lerp([(0, 1240, 560), (12, 40, 560)])
    s = run(model, [(1, 2, a), (2, 2, b), (3, 2, c)])
    assert s.max() < 0.3


def test_scores_are_causal(model):
    a = lerp([(0, 40, 420), (5.0, 900, 420), (5.4, 950, 420), (12, 950, 420)])
    b = lerp([(0, 1040, 420), (12, 1100, 420)])
    full = run(model, [(1, 2, a), (2, 2, b)], seconds=12.0)
    part = run(model, [(1, 2, a), (2, 2, b)], seconds=6.0)
    assert np.allclose(full[: len(part)], part)
