"""Risk scores for a prefix must not change when later frames are never fed."""

import cv2
import numpy as np

import solution


def _scores(clip, n_frames):
    est = solution.RiskEstimator()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    est.reset({"video_id": clip.name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": int(cap.get(7))})
    out = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        out.append(est.step(frame, i / fps))
    cap.release()
    return np.array(out)


def test_prefix_scores_do_not_depend_on_future(clip):
    full = _scores(clip, 10_000)
    n = len(full) // 2
    prefix = _scores(clip, n)
    assert len(prefix) == n
    assert np.allclose(full[:n], prefix, atol=1e-6)


def test_reset_clears_state(clip):
    a = _scores(clip, 40)
    b = _scores(clip, 40)
    assert np.allclose(a, b, atol=1e-6)
