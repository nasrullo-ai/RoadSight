"""EventPipeline (Part A): video -> detections -> tracks -> kinematics -> scene -> rules -> segments."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from roadsight import CLASSES
from roadsight.config import load_config
from roadsight.events import rules as _rules  # noqa: F401  (registers every rule)
from roadsight.events.base import RULES, Segment
from roadsight.events.postprocess import postprocess
from roadsight.io.video import FrameReader, VideoMeta
from roadsight.perception.detector import Detector, select_device
from roadsight.perception.signal import SignalTimeline, classify_roi
from roadsight.perception.tracker import Tracker
from roadsight.perception.tracktable import TrackTable
from roadsight.scene.auto import AutoScene
from roadsight.scene.scene import Scene
from roadsight.utils.log import get_logger
from roadsight.utils.seed import seed_everything
from roadsight.utils.timing import StageTimer

log = get_logger("roadsight.pipeline")

Progress = Callable[[float, str], None]


@dataclass
class PipelineResult:
    events: list[list]
    segments: list[Segment]
    tracks: TrackTable
    scene: Scene
    signal: SignalTimeline
    meta: VideoMeta
    timing: dict = field(default_factory=dict)
    stride: int = 1


class EventPipeline:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = select_device(cfg.get("device", "auto"))
        self._detector: Detector | None = None

    @classmethod
    def from_config(cls, path: str = "configs/default.yaml", overrides: dict | None = None) -> EventPipeline:
        return cls(load_config(path, overrides))

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector(self.cfg["detector"], self.device)
            self._detector.warmup()
        return self._detector

    def run(self, video_path: str, progress: Progress | None = None) -> list[list]:
        return self.run_full(video_path, progress).events

    # ------------------------------------------------------------------ main loop
    def run_full(self, video_path: str, progress: Progress | None = None) -> PipelineResult:
        cfg = self.cfg
        seed_everything(int(cfg.get("seed", 0)))
        timer = StageTimer()
        vcfg = cfg.get("video", {})
        on_cpu = self.device == "cpu"
        stride = int(vcfg.get("stride_cpu", 5) if on_cpu else vcfg.get("stride", 2))
        detector = self.detector

        reader = FrameReader(video_path, stride)
        meta = reader.meta
        scene = Scene.load(cfg.get("scene", {}).get("path"), meta.width, meta.height)
        tracker = Tracker(cfg.get("tracker", {}))

        ev_cfg = cfg.get("events", {})
        want_bg = ev_cfg.get("road_obstacle", {}).get("enabled", False) and ev_cfg.get("road_obstacle", {}).get("blob_enabled", False)
        want_color = ev_cfg.get("fire_smoke", {}).get("enabled", False)
        thumb_every = float(vcfg.get("thumb_interval_sec", 1.0))
        thumb_w = int(vcfg.get("thumb_width", 160))

        cache = self._cache_path(video_path, stride)
        if cache is not None and cache.exists():
            with timer.stage("cache-load"):
                records, frame_ids, frame_times, sig_samples, final_stride, n_decoded = self._load_cache(cache)
            reader.stride = final_stride
            meta.n_frames = n_decoded or meta.n_frames
            bg_samples, color_samples = [], []
        else:
            records: list[np.ndarray] = []
            frame_ids: list[int] = []
            frame_times: list[float] = []
            sig_samples: dict[str, list[tuple[float, str]]] = {r.id: [] for r in scene.signal_rois}
            bg_samples: list[tuple[float, np.ndarray]] = []
            color_samples: list[tuple[float, np.ndarray]] = []
            next_thumb = 0.0

            batch_frames: list[np.ndarray] = []
            batch_info: list[tuple[int, float]] = []
            # Adaptive stride only fires far above the expected speed, so normal runs stay deterministic.
            adaptive = bool(vcfg.get("adaptive", True))
            trigger_rtf = float(vcfg.get("adaptive_trigger_rtf", 1.0))
            goal_rtf = float(vcfg.get("adaptive_goal_rtf", 0.6))
            max_stride = int(vcfg.get("max_stride", 4))
            probe_frames = int(vcfg.get("probe_frames", 200))
            t_start = time.perf_counter()
            n_total = max(1, meta.n_frames)

            def flush() -> None:
                if not batch_frames:
                    return
                with timer.stage("detect"):
                    dets = detector(batch_frames)
                with timer.stage("track"):
                    for (fidx, ft), d in zip(batch_info, dets):
                        tr = tracker.update(d)
                        frame_ids.append(fidx)
                        frame_times.append(ft)
                        if len(tr):
                            rec = np.empty((len(tr), 9))
                            rec[:, 0] = fidx
                            rec[:, 1] = ft
                            rec[:, 2] = tr[:, 4]
                            rec[:, 3] = tr[:, 6]
                            rec[:, 4] = tr[:, 5]
                            rec[:, 5:9] = tr[:, :4]
                            records.append(rec)
                batch_frames.clear()
                batch_info.clear()

            with timer.stage("decode+loop"):
                for fidx, ft, frame in reader:
                    for roi in scene.signal_rois:
                        sig_samples[roi.id].append((ft, classify_roi(frame, roi.rect)))
                    if (want_bg or want_color) and ft >= next_thumb:
                        h = int(round(frame.shape[0] * thumb_w / frame.shape[1]))
                        small = cv2.resize(frame, (thumb_w, h), interpolation=cv2.INTER_AREA)
                        if want_bg:
                            bg_samples.append((ft, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
                        if want_color:
                            color_samples.append((ft, small))
                        next_thumb = ft + thumb_every
                    batch_frames.append(frame)
                    batch_info.append((fidx, ft))
                    if len(batch_frames) >= detector.batch:
                        flush()
                        if progress:
                            progress(min(0.95, fidx / n_total), "detecting")
                    if adaptive and fidx >= probe_frames and fidx < probe_frames + reader.stride:
                        elapsed = time.perf_counter() - t_start
                        rtf = elapsed / max(ft, 1e-3)
                        if rtf > trigger_rtf and reader.stride < max_stride:
                            new = min(max_stride, int(np.ceil(reader.stride * rtf / goal_rtf)))
                            log.info("adaptive stride %d -> %d (projected Part A RTF %.2f)", reader.stride, new, rtf)
                            reader.stride = new
                flush()
            if reader.frames_decoded > 0:
                meta.n_frames = reader.frames_decoded
            if cache is not None:
                self._save_cache(cache, records, frame_ids, frame_times, sig_samples, reader.stride, meta.n_frames)

        with timer.stage("tracks"):
            recs = np.concatenate(records) if len(records) else np.zeros((0, 9))
            tcfg = {**cfg.get("tracker", {}), **cfg.get("kinematics", {})}
            tracks = TrackTable.from_records(recs, np.array(frame_times), np.array(frame_ids), meta.width, meta.height, tcfg)
            tracks.bg_samples = bg_samples
            tracks.color_samples = color_samples
        with timer.stage("scene"):
            scene.auto = AutoScene.build(tracks.df, meta.width, meta.height, cfg.get("scene", {}))
            signal = SignalTimeline.from_samples(sig_samples, float(cfg.get("signal", {}).get("smooth_sec", 1.0)))

        segments: list[Segment] = []
        with timer.stage("rules"):
            for label in CLASSES:
                c = ev_cfg.get(label, {}) or {}
                if not c.get("enabled", False) or label not in RULES:
                    continue
                try:
                    segs = RULES[label](c, scene).detect(tracks, signal, meta)
                    segments.extend(segs)
                except Exception as exc:  # a broken rule must never sink the video
                    log.warning("rule %s failed: %r", label, exc)
            events = postprocess(segments, meta.duration, ev_cfg)
        if progress:
            progress(1.0, "done")
        timer.log(f"{meta.video_id} ")
        return PipelineResult(events, segments, tracks, scene, signal, meta, timer.summary(), reader.stride)

    # ------------------------------------------------------------------ dev-only perception cache
    def _cache_path(self, video_path: str, stride: int) -> Path | None:
        """Enabled only by ROADSIGHT_CACHE_DIR (tools/tune.py); never used by the submission."""
        root = os.environ.get("ROADSIGHT_CACHE_DIR")
        if not root:
            return None
        st = os.stat(video_path)
        key = json.dumps(
            [
                Path(video_path).name,
                st.st_size,
                stride,
                self.device,
                self.cfg["detector"],
                self.cfg.get("tracker", {}),
                self.cfg.get("scene", {}).get("path"),
            ],
            sort_keys=True,
            default=str,
        )
        return Path(root) / f"{Path(video_path).stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"

    @staticmethod
    def _save_cache(path: Path, records, frame_ids, frame_times, sig_samples, stride, n_decoded) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        recs = np.concatenate(records) if records else np.zeros((0, 9))
        np.savez_compressed(
            path,
            records=recs,
            frame_ids=np.asarray(frame_ids),
            frame_times=np.asarray(frame_times),
            signal=json.dumps(sig_samples),
            stride=stride,
            n_decoded=n_decoded,
        )

    @staticmethod
    def _load_cache(path: Path):
        z = np.load(path, allow_pickle=False)
        sig = {k: [(float(a), str(b)) for a, b in v] for k, v in json.loads(str(z["signal"])).items()}
        return [z["records"]], list(z["frame_ids"]), list(z["frame_times"]), sig, int(z["stride"]), int(z["n_decoded"])
