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
        self.scale = (1.0, 1.0)

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


class FFmpegReader:
    """Decode with the bundled ffmpeg binary: multi-threaded decoding, frame selection and downscaling
    all happen inside ffmpeg, so only the processed frames are converted, at a reduced size.

    With ``skip_nonref`` the decoder skips non-reference (B) frames entirely, which on long-GOP camera
    footage (IBBP) cuts decoding by ~2.5x. Frames are then picked by timestamp (at most ``target_hz``)
    and each frame's source index comes from its presentation time (``showinfo``), so timing stays
    exact whatever the GOP structure. ``scale`` maps output pixels back to the original resolution.
    """

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        max_width: int = 1920,
        skip_nonref: bool = True,
        target_hz: float | None = None,
        crops: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.meta = probe(path)
        # Full-resolution crops (x1, y1, x2, y2 in original pixels) delivered with every frame in
        # ``last_crops``, e.g. traffic-signal heads that would blur away in the downscaled frame.
        self.crops = [self._clamp(c) for c in (crops or [])]
        self.crops = [c for c in self.crops if c[2] > c[0] and c[3] > c[1]]
        self.last_crops: list[np.ndarray] | None = [] if self.crops else None
        self.stride = max(1, int(stride))
        self.skip_nonref = skip_nonref
        self.target_hz = target_hz
        self.frames_decoded = 0
        w, h = self.meta.width, self.meta.height
        out_w = min(int(max_width), w) if w > 0 else int(max_width)
        out_w -= out_w % 2
        out_h = int(round(h * out_w / max(w, 1) / 2.0)) * 2
        self.size = (out_w, out_h)
        self.scale = (w / out_w if out_w else 1.0, h / out_h if out_h else 1.0)

    def _clamp(self, c) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(v) for v in c)
        x1, y1 = max(0, x1 - x1 % 2), max(0, y1 - y1 % 2)
        x2, y2 = min(self.meta.width, x2 + x2 % 2), min(self.meta.height, y2 + y2 % 2)
        return x1, y1, x2, y2

    @property
    def strip_height(self) -> int:
        return max((c[3] - c[1] for c in self.crops), default=0)

    def _command(self) -> list[str]:
        import imageio_ffmpeg

        out_w, out_h = self.size
        hz = self.target_hz or self.meta.fps / self.stride
        min_gap = 0.9 / hz  # keep frames at most ~hz apart; 0.9 tolerates timestamp jitter
        select = rf"select=isnan(prev_selected_t)+gte(t-prev_selected_t\,{min_gap:.4f})"
        dec = ["-skip_frame", "noref"] if self.skip_nonref else []
        if not self.crops:
            graph = ["-vf", f"{select},showinfo,scale={out_w}:{out_h}:flags=area"]
        else:
            # One output frame: the downscaled picture with the full-resolution crops stacked below it.
            n = len(self.crops)
            sh = self.strip_height
            parts = [f"[0:v]{select},showinfo,split={n + 1}[m]" + "".join(f"[s{i}]" for i in range(n))]
            parts.append(f"[m]scale={out_w}:{out_h}:flags=area[main]")
            for i, (x1, y1, x2, y2) in enumerate(self.crops):
                parts.append(f"[s{i}]crop={x2 - x1}:{y2 - y1}:{x1}:{y1},pad={x2 - x1}:{sh}:0:0[c{i}]")
            row = "".join(f"[c{i}]" for i in range(n))
            parts.append(f"{row}hstack=inputs={n}[cs]" if n > 1 else "[c0]null[cs]")
            parts.append(f"[cs]pad={out_w}:{sh}:0:0[strip]")
            parts.append("[main][strip]vstack[out]")
            graph = ["-filter_complex", ";".join(parts), "-map", "[out]"]
        return [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-nostdin",
            "-threads",
            "0",
            *dec,
            "-i",
            self.meta.path,
            "-an",
            "-sn",
            "-dn",
            *graph,
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import queue
        import re
        import subprocess
        import threading

        out_w, out_h = self.size
        if out_w <= 0 or out_h <= 0:
            raise OSError(f"cannot read video size: {self.meta.path}")
        total_h = out_h + (self.strip_height if self.crops else 0)
        widths = [c[2] - c[0] for c in self.crops]
        if sum(widths) > out_w:
            raise ValueError("signal crops wider than the frame")
        n_bytes = out_w * total_h * 3
        proc = subprocess.Popen(self._command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=n_bytes * 2)
        pts: queue.Queue = queue.Queue()
        pat = re.compile(rb"Parsed_showinfo.*?pts_time:\s*(-?[0-9.]+)")

        def pump() -> None:
            for line in proc.stderr:
                m = pat.search(line)
                if m:
                    pts.put(float(m.group(1)))
            pts.put(None)

        threading.Thread(target=pump, daemon=True).start()
        fps = self.meta.fps
        last_idx = -1
        try:
            while True:
                buf = proc.stdout.read(n_bytes)
                if len(buf) < n_bytes:
                    break
                t = pts.get(timeout=60)
                if t is None:
                    break
                idx = max(last_idx + 1, int(round(max(t, 0.0) * fps)))
                last_idx = idx
                arr = np.frombuffer(buf, np.uint8).reshape(total_h, out_w, 3)
                if self.crops:
                    x = 0
                    self.last_crops = []
                    for (_x1, y1, _x2, y2), w in zip(self.crops, widths):
                        self.last_crops.append(arr[out_h : out_h + (y2 - y1), x : x + w])
                        x += w
                yield idx, idx / fps, arr[:out_h]
        finally:
            proc.kill()
            proc.wait()
            self.frames_decoded = max(self.meta.n_frames, last_idx + 1)
