import json

import evaluate
import run_submission


def test_harness_output_validates(clip_dir, tmp_path):
    out = tmp_path / "predictions.json"
    assert run_submission.main(["--videos", str(clip_dir), "--out", str(out)]) == 0
    pred = json.loads(out.read_text())
    assert set(pred["videos"]) == {p.name for p in clip_dir.glob("*.mp4")}
    assert evaluate.validate(pred) == []
    assert evaluate.main(["--pred", str(out), "--validate-only"]) == 0


def test_perfect_prediction_scores_one():
    gt = {"videos": {"a.mp4": {"duration": 60, "fps": 25, "events": [[1.0, 5.0, "jaywalking"], [10.0, 30.0, "congestion"]]}}}
    pred = {"team": "t", "videos": {"a.mp4": {"events": gt["videos"]["a.mp4"]["events"], "risk": []}}}
    assert evaluate.score_a(pred, gt)["score_a"] == 1.0


def test_spurious_class_adds_zero():
    gt = {"videos": {"a.mp4": {"duration": 60, "fps": 25, "events": [[1.0, 5.0, "jaywalking"]]}}}
    pred = {"team": "t", "videos": {"a.mp4": {"events": [[1.0, 5.0, "jaywalking"], [7.0, 9.0, "fire_smoke"]], "risk": []}}}
    assert evaluate.score_a(pred, gt)["score_a"] == 0.5
