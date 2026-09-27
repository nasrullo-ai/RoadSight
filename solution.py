"""Submission interface (SPEC section 3). Thin adapter; all logic lives in the ``roadsight`` package."""

import os
import traceback

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from roadsight.pipeline import EventPipeline  # noqa: E402
from roadsight.risk import CausalRiskModel  # noqa: E402

CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

# Anticipation horizon used by the metric: step() returns P(an accident starts within this many seconds).
RISK_HORIZON_SEC = 5.0

CONFIG = os.environ.get("ROADSIGHT_CONFIG", "configs/default.yaml")

_pipeline = None  # lazy singleton: load weights once per process


def detect_events(video_path: str) -> list[list]:
    global _pipeline
    try:
        if _pipeline is None:
            _pipeline = EventPipeline.from_config(CONFIG)
        return _pipeline.run(video_path)  # [[float, float, str], ...]
    except Exception:
        traceback.print_exc()  # stderr only; never raise
        return []


class RiskEstimator:
    """Part B. Never raises: if the model cannot load or reset, it returns the floor score (0.0 before any model)."""

    def __init__(self):
        self._ok = False
        try:
            self._model = CausalRiskModel.from_config(CONFIG)
        except Exception:
            self._model = None

    def reset(self, meta: dict) -> None:
        self._ok = False
        if self._model is None:
            return
        try:
            self._model.reset(meta)
            self._ok = True
        except Exception:
            pass

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if not self._ok:
            return 0.0
        try:
            return float(min(1.0, max(0.0, self._model.step(frame, t_sec))))
        except Exception:
            return float(self._model.last_score)
