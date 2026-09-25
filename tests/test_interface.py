import cv2
import numpy as np

import roadsight
import solution


def test_classes_match_spec():
    assert solution.CLASSES == roadsight.CLASSES
    assert len(solution.CLASSES) == 14


def test_detect_events_returns_valid_list(clip):
    events = solution.detect_events(str(clip))
    assert isinstance(events, list)
    for ev in events:
        assert isinstance(ev, list) and len(ev) == 3
        s, e, lab = ev
        assert isinstance(s, float) and isinstance(e, float)
        assert 0.0 <= s < e
        assert lab in solution.CLASSES


def test_detect_events_never_raises_on_missing_file(tmp_path):
    assert solution.detect_events(str(tmp_path / "missing.mp4")) == []


def test_risk_estimator_scores_in_unit_interval(clip):
    est = solution.RiskEstimator()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    est.reset({"video_id": clip.name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": int(cap.get(7))})
    scores = []
    for i in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(est.step(frame, i / fps))
    cap.release()
    assert scores and all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores)


def test_risk_estimator_survives_bad_frame():
    est = solution.RiskEstimator()
    est.reset({"video_id": "x", "fps": 25.0, "width": 64, "height": 64, "n_frames": 3})
    s = est.step(np.zeros((0, 0, 3), np.uint8), 0.0)
    assert 0.0 <= s <= 1.0
