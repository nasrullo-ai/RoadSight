"""The organizers' harness and evaluator, run exactly as they will run them."""

import json
import subprocess
import sys
from pathlib import Path

import evaluate

ROOT = Path(__file__).resolve().parent.parent


def test_harness_output_validates(clip_dir, tmp_path):
    out = tmp_path / "predictions.json"
    run = subprocess.run(
        [sys.executable, "run_submission.py", "--videos", str(clip_dir), "--out", str(out), "--team", "roadsight"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    pred = json.loads(out.read_text())
    assert set(pred["videos"]) == {p.name for p in clip_dir.glob("*.mp4")}
    for vid, log in pred["log"].items():
        assert log["errors"] == [], (vid, log["errors"])  # no drops, no exceptions, inside the time budget
    errors, _ = evaluate.validate(pred)
    assert errors == []
    check = subprocess.run([sys.executable, "evaluate.py", "--pred", str(out), "--validate-only"], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0 and "VALID" in check.stdout, check.stdout


def test_perfect_prediction_scores_one():
    gt = {"a.mp4": {"duration": 60, "fps": 25, "events": [[1.0, 5.0, "jaywalking"], [10.0, 30.0, "congestion"]]}}
    pred = {"a.mp4": {"events": gt["a.mp4"]["events"], "risk": []}}
    assert evaluate.evaluate_part_a(gt, pred)["score_a"] == 1.0


def test_spurious_class_adds_zero():
    gt = {"a.mp4": {"duration": 60, "fps": 25, "events": [[1.0, 5.0, "jaywalking"]]}}
    pred = {"a.mp4": {"events": [[1.0, 5.0, "jaywalking"], [7.0, 9.0, "fire_smoke"]], "risk": []}}
    assert evaluate.evaluate_part_a(gt, pred)["score_a"] == 0.5


def test_committed_sample_predictions_validate():
    pred = json.loads((ROOT / "predictions_samples.json").read_text())
    errors, _ = evaluate.validate(pred)
    assert errors == []
