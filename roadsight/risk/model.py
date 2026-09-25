"""CausalRiskModel (Part B): its own detector + tracker, fed frame by frame; no look-ahead, no file access.

score_t = sigmoid(w0 + sum_i w_i * f_i,t), then causal EMA, a 1 s peak-hold and a density floor.
The detector runs on every ``stride``-th frame; other frames return the cached score.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np

from roadsight.config import load_config, resolve_path
from roadsight.perception.detector import Detector, select_device
from roadsight.perception.kinematics import VEHICLE_CLASSES, OnlineKinematics, RowScale
from roadsight.perception.signal import RED, UNKNOWN, classify_roi
from roadsight.perception.tracker import Tracker
from roadsight.risk.features import FEATURES, measure
from roadsight.scene.auto import AutoScene
from roadsight.scene.geometry import side_of_polyline
from roadsight.scene.scene import Scene
from roadsight.utils.seed import seed_everything

DEFAULT_WEIGHTS = {
    "bias": -5.0,
    "f_ttc": 2.0,
    "f_brake": 1.0,
    "f_swerve": 1.0,
    "f_conflict": 3.0,
    "f_wrong": 2.5,
    "f_redrun": 2.0,
    "f_ped": 2.0,
    "f_density": 0.5,
}


class CausalRiskModel:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.rcfg = cfg.get("risk", {})
        self.device = select_device(cfg.get("device", "auto"))
        self.weights = self._load_weights()
        self._detector: Detector | None = None
        self.last_score = float(self.rcfg.get("floor", 0.01))
        self.last_features: dict[str, float] = dict.fromkeys(FEATURES, 0.0)
        self.meta: dict = {}

    @classmethod
    def from_config(cls, path: str = "configs/default.yaml", overrides: dict | None = None) -> CausalRiskModel:
        return cls(load_config(path, overrides))

    def _load_weights(self) -> dict[str, float]:
        w = dict(DEFAULT_WEIGHTS)
        w.update(self.rcfg.get("weights", {}) or {})
        p = self.rcfg.get("weights_file")
        if p and resolve_path(p).exists():
            with open(resolve_path(p), encoding="utf-8") as f:
                w.update(json.load(f))
        return w

    # ------------------------------------------------------------------ lifecycle
    def reset(self, meta: dict) -> None:
        seed_everything(int(self.cfg.get("seed", 0)))
        if self._detector is None:  # lazy: load weights on the first reset only
            dcfg = dict(self.cfg["detector"])
            if self.rcfg.get("detector_weights"):
                dcfg["weights"] = self.rcfg["detector_weights"]
            if self.rcfg.get("imgsz"):
                dcfg["imgsz"] = self.rcfg["imgsz"]
            dcfg["batch"] = 1
            self._detector = Detector(dcfg, self.device)
            self._detector.warmup()
        self.meta = dict(meta)
        w, h = int(meta.get("width", 1920) or 1920), int(meta.get("height", 1080) or 1080)
        self.fps = float(meta.get("fps", 25.0) or 25.0)
        on_cpu = self.device == "cpu"
        self.stride = max(1, int(self.rcfg.get("stride_cpu", 6) if on_cpu else self.rcfg.get("stride", 3)))
        self.tracker = Tracker(self.cfg.get("tracker", {}))
        self.kin = OnlineKinematics(rate_hz=self.fps / self.stride, width=w, height=h)
        self.height = h
        self._noise = {"decel": deque(maxlen=600), "swerve": deque(maxlen=600)}
        self.scale = RowScale.default(h)
        self._scale_y: deque = deque(maxlen=4000)
        self._scale_s: deque = deque(maxlen=4000)
        self.scene = Scene.load(self.cfg.get("scene", {}).get("path"), w, h)
        self.auto = AutoScene.empty(w, h, self.cfg.get("scene", {}))
        self.scene.auto = self.auto
        self._road_counts = np.zeros_like(self.auto.road, dtype=np.float64)
        self._sig_hist = {r.id: deque(maxlen=max(1, int(self.fps / self.stride))) for r in self.scene.signal_rois}
        self._line_sides: dict[str, float] = {}
        self.n = 0
        self.n_processed = 0
        self.ema: float | None = None
        self._hold: deque = deque()
        self.last_score = float(self.rcfg.get("floor", 0.01))
        self.last_features = dict.fromkeys(FEATURES, 0.0)

    # ------------------------------------------------------------------ per frame
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        k = self.n
        self.n += 1
        if k % self.stride != 0:
            return self.last_score
        det = self._detector([frame])[0]
        tracks = self.tracker.update(det)
        self._update_scale(tracks)
        self.kin.update(t_sec, tracks, self.scale)
        self._update_scene(t_sec)
        red_lines = self._red_lines(frame)
        active = self.kin.active(t_sec)
        raw = measure(active, t_sec, self.scale, self.scene, red_lines, self.rcfg, self.height)
        feats = self._normalise(raw)
        self.last_raw = raw
        self.last_features = feats
        self.n_processed += 1
        self.last_score = self._combine(feats, t_sec)
        return self.last_score

    def _normalise(self, raw: dict[str, float]) -> dict[str, float]:
        """Raw measurements -> 0..1 features. Braking and swerving are scaled by the larger of a fixed
        reference and ``noise_k`` x the running median of recent values (causal noise floor)."""
        k = float(self.rcfg.get("noise_k", 3.0))
        f = {}
        ttc = raw["ttc"]
        f["f_ttc"] = float(np.exp(-ttc / float(self.rcfg.get("ttc_tau", 1.5)))) if ttc < float(self.rcfg.get("ttc_max", 2.0)) else 0.0
        for key, name, ref in (("decel", "f_brake", "brake_ref"), ("swerve", "f_swerve", "swerve_ref_deg_s")):
            hist = self._noise[key]
            hist.append(raw[key])
            floor = k * float(np.median(hist)) if len(hist) >= 20 else 0.0
            f[name] = float(np.clip(raw[key] / max(float(self.rcfg.get(ref, 1.0)), floor, 1e-6), 0.0, 1.0))
        f["f_conflict"] = f["f_ttc"] * max(f["f_brake"], f["f_swerve"])  # closing in AND evading
        f["f_wrong"] = raw["wrong"]
        f["f_redrun"] = raw["redrun"]
        f["f_ped"] = raw["ped"]
        f["f_density"] = float(min(1.0, raw["n_veh"] / float(self.rcfg.get("density_norm", 20))))
        return f

    def _combine(self, feats: dict[str, float], t: float) -> float:
        z = self.weights["bias"] + sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        r = 1.0 / (1.0 + np.exp(-z))
        alpha = float(self.rcfg.get("ema_alpha", 0.3))
        self.ema = r if self.ema is None else self.ema + alpha * (r - self.ema)
        hold = float(self.rcfg.get("peak_hold_sec", 1.0))
        self._hold.append((t, self.ema))
        while self._hold and self._hold[0][0] < t - hold:
            self._hold.popleft()
        out = max(v for _, v in self._hold)
        floor = float(self.rcfg.get("floor", 0.01)) + float(self.rcfg.get("floor_density", 0.04)) * feats["f_density"]
        return float(np.clip(max(out, floor), 0.0, 1.0))

    # ------------------------------------------------------------------ online scene state
    def _update_scale(self, tracks: np.ndarray) -> None:
        if len(tracks) == 0:
            return
        veh = tracks[np.isin(tracks[:, 6], VEHICLE_CLASSES)]
        for x1, y1, x2, y2 in veh[:, :4]:
            self._scale_y.append(y2)
            self._scale_s.append(np.sqrt(max(1.0, x2 - x1) * max(1.0, y2 - y1)))
        if self.n_processed % 50 == 0 and len(self._scale_s) >= 30:
            self.scale = RowScale.fit(np.array(self._scale_y), np.array(self._scale_s), self.auto.grid.height)

    def _update_scene(self, t: float) -> None:
        xs, ys, ux, uy = [], [], [], []
        for _, st in self.kin.states.items():
            if st.cls not in VEHICLE_CLASSES or st.t != t or st.edge:
                continue
            iy, ix = self.auto.grid.index(st.x, st.y)
            self._road_counts[iy, ix] += 1
            sp = np.hypot(st.vx, st.vy)
            if sp > 0.5 and st.n >= 3:
                xs.append(st.x)
                ys.append(st.y)
                ux.append(st.vx / sp)
                uy.append(st.vy / sp)
        if xs:
            self.auto.flow.add_samples(np.array(xs), np.array(ys), np.array(ux), np.array(uy))
        if self.n_processed % 25 == 0:
            self.auto.road = self._road_counts >= float(self.rcfg.get("road_min_obs", 5))

    def _red_lines(self, frame: np.ndarray) -> list:
        if not self.scene.stop_lines or not self.scene.signal_rois:
            return []
        state = {}
        for roi in self.scene.signal_rois:
            self._sig_hist[roi.id].append(classify_roi(frame, roi.rect))
            hist = [s for s in self._sig_hist[roi.id] if s != UNKNOWN]
            state[roi.id] = max(set(hist), key=hist.count) if hist else UNKNOWN
        out = []
        for line in self.scene.stop_lines:
            sid = line.signal or (self.scene.signal_rois[0].id if len(self.scene.signal_rois) == 1 else None)
            if sid is None or state.get(sid) != RED:
                continue
            side = self._line_sides.get(line.id)
            if side is None:
                if line.approach is None:
                    continue
                mid = line.points.mean(0) - line.approach * 10.0
                side = float(side_of_polyline(mid[:1], mid[1:], line.points)[0])
                self._line_sides[line.id] = side
            out.append((line, side))
        return out


def run_risk(video_path: str, model: CausalRiskModel, progress=None, every: int = 1) -> list[list[float]]:
    """Feed a whole video to ``model`` frame by frame, as the harness does; returns [[t, score], ...]."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if 1.0 <= fps <= 240.0 else 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model.reset({"video_id": video_path, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)), "n_frames": n})
    out = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            s = model.step(frame, i / fps)
            if i % every == 0:
                out.append([round(i / fps, 2), round(float(s), 4)])
            i += 1
            if progress and i % 50 == 0 and n:
                progress(min(1.0, i / n))
    finally:
        cap.release()
    return out
