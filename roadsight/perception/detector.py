"""YOLO detector wrapper: FP16 batched inference, offline weights, one model per process."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from roadsight.config import resolve_path
from roadsight.utils.log import get_logger

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

log = get_logger("roadsight.detector")

_MODELS: dict[tuple[str, str], object] = {}


def select_device(pref: str = "auto") -> str:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if pref == "cpu" or not has_cuda:
        return "cpu"
    return "cuda:0" if pref in ("auto", "cuda") else pref


def load_model(weights: str | Path, device: str):
    """Load (and cache) an Ultralytics model from a local file. Never downloads."""
    path = resolve_path(weights)
    key = (str(path), device)
    if key not in _MODELS:
        if not path.exists():
            raise FileNotFoundError(f"detector weights missing: {path}")
        from ultralytics import YOLO

        model = YOLO(str(path), task="detect")
        _MODELS[key] = model
        log.info("loaded %s on %s", path.name, device)
    return _MODELS[key]


class Detector:
    """Returns one (N, 6) array per frame: x1, y1, x2, y2, conf, cls (original pixel coordinates)."""

    def __init__(self, cfg: dict, device: str) -> None:
        self.device = device
        on_cpu = device == "cpu"
        weights = cfg.get("weights_cpu", cfg["weights"]) if on_cpu else cfg["weights"]
        self.model = load_model(weights, device)
        self.imgsz = int(cfg.get("imgsz_cpu", cfg.get("imgsz", 640)) if on_cpu else cfg.get("imgsz", 640))
        self.half = bool(cfg.get("half", True)) and not on_cpu
        self.batch = int(cfg.get("batch", 16))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.5))
        self.classes = sorted(set(cfg.get("classes", [0, 1, 2, 3, 5, 7])) | set(cfg.get("extra_classes", [])))
        self._warm = False

    def warmup(self, shape: tuple[int, int, int] = (640, 640, 3)) -> None:
        if not self._warm:
            self(np.zeros(shape, dtype=np.uint8)[None])
            self._warm = True

    def __call__(self, frames) -> list[np.ndarray]:
        frames = list(frames)
        if not frames:
            return []
        out: list[np.ndarray] = []
        for i in range(0, len(frames), self.batch):
            chunk = frames[i : i + self.batch]
            try:
                results = self._predict(chunk)
            except Exception as exc:
                if self.device == "cpu":
                    raise
                # A broken CUDA stack (e.g. CPU-only torchvision NMS) would otherwise empty every video.
                log.error("GPU inference failed (%r); falling back to CPU", exc)
                self.device, self.half = "cpu", False
                results = self._predict(chunk)
            for r in results:
                data = r.boxes.data
                out.append(data.float().cpu().numpy()[:, :6] if len(data) else np.zeros((0, 6), dtype=np.float32))
        return out

    def _predict(self, chunk):
        return self.model.predict(
            chunk,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            device=self.device,
            batch=len(chunk),
            verbose=False,
        )
