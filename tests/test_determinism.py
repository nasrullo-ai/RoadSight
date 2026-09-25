import cv2
import numpy as np

import solution


def _risk(clip):
    est = solution.RiskEstimator()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    est.reset({"video_id": clip.name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": int(cap.get(7))})
    out = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(est.step(frame, i / fps))
        i += 1
    cap.release()
    return np.array(out)


def test_two_runs_identical_events(clip):
    a = solution.detect_events(str(clip))
    b = solution.detect_events(str(clip))
    assert a == b


def test_two_runs_identical_risk(clip):
    a = _risk(clip)
    b = _risk(clip)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-6)
