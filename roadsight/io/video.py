"""Sequential video reading with a processing stride.

Seeking is slow and non-deterministic on some codecs, so frames are always decoded in order;
skipped frames are only grabbed, not converted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 25.0


@dataclass
class VideoMeta:
    path: str
    video_id: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
            "duration": round(self.duration, 3),
        }


def stride_for(fps: float, target_hz: float | None, fallback: int) -> int:
    """Frames to step so processing runs near ``target_hz`` whatever the video frame rate."""
    if not target_hz or target_hz <= 0:
        return max(1, int(fallback))
    return max(1, int(round(sanitize_fps(fps) / float(target_hz))))


def sanitize_fps(fps: float) -> float:
    return float(fps) if fps and 1.0 <= fps <= 240.0 else DEFAULT_FPS


def probe(path: str | Path) -> VideoMeta:
    """Read container metadata without decoding frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        fps = sanitize_fps(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    return VideoMeta(str(path), Path(path).name, fps, width, height, n_frames)


class FrameReader:
    """Yields ``(frame_idx, t_sec, frame_bgr)`` for every ``stride``-th frame.

    ``stride`` may be changed while iterating (adaptive stride). After iteration,
    ``frames_decoded`` holds the true number of frames in the file.
    """

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        self.meta = probe(path)
        self.stride = max(1, int(stride))
        self.frames_decoded = 0

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.meta.path)
        if not cap.isOpened():
            raise OSError(f"cannot open video: {self.meta.path}")
        fps = self.meta.fps
        idx = 0
        next_idx = 0
        try:
            while True:
                if not cap.grab():
                    break
                if idx == next_idx:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        yield idx, idx / fps, frame
                    next_idx = idx + self.stride
                idx += 1
        finally:
            self.frames_decoded = idx
            cap.release()
