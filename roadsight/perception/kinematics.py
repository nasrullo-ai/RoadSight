"""Speed, heading and acceleration per track.

Speeds are expressed in body-lengths per second (BL/s): pixel speed divided by the typical
vehicle size at that image row, so thresholds transfer across the perspective of the image.
Part A smooths with Savitzky-Golay (non-causal); Part B uses causal EMA (``OnlineKinematics``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

VEHICLE_CLASSES = (2, 3, 5, 7)


@dataclass
class RowScale:
    """Typical vehicle size (px) as a linear function of the foot-point row."""

    a: float
    b: float
    min_px: float

    def __call__(self, y) -> np.ndarray:
        return np.maximum(self.min_px, self.a + self.b * np.asarray(y, dtype=np.float64))

    @classmethod
    def default(cls, height: int) -> RowScale:
        return cls(a=0.08 * height, b=0.0, min_px=0.02 * height)

    @classmethod
    def fit(cls, foot_y: np.ndarray, size: np.ndarray, height: int, min_samples: int = 30) -> RowScale:
        """Robust fit from vehicle boxes: per-row-bin medians, then a weighted line."""
        foot_y = np.asarray(foot_y, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        if len(size) < min_samples:
            return cls.default(height) if len(size) == 0 else cls(float(np.median(size)), 0.0, 0.25 * float(np.median(size)))
        med_all = float(np.median(size))
        edges = np.linspace(foot_y.min(), foot_y.max() + 1e-6, 11)
        centers, meds, weights = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (foot_y >= lo) & (foot_y < hi)
            if m.sum() >= 5:
                centers.append((lo + hi) / 2)
                meds.append(np.median(size[m]))
                weights.append(np.sqrt(m.sum()))
        if len(centers) < 2:
            return cls(med_all, 0.0, 0.25 * med_all)
        b, a = np.polyfit(centers, meds, 1, w=weights)
        if b < 0:  # perspective should never shrink objects towards the camera
            return cls(med_all, 0.0, 0.25 * med_all)
        return cls(float(a), float(b), 0.25 * med_all)


def odd_window(n_samples: int, want: int) -> int:
    w = min(want, n_samples if n_samples % 2 == 1 else n_samples - 1)
    return w if w >= 5 else 0


def smooth_track(
    t: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    scale: RowScale,
    window_sec: float = 1.0,
    order: int = 2,
    ax: np.ndarray | None = None,
    ay: np.ndarray | None = None,
):
    """Return sx, sy, vx, vy (BL/s), speed (BL/s), heading (deg), accel (BL/s^2).

    ``fx, fy`` is the ground point used for positions; ``ax, ay`` (optional) is the anchor used
    for velocity -- the box top, which is rarely occluded by closer traffic.
    """
    n = len(t)
    if n < 2:
        z = np.zeros(n)
        return fx.astype(float), fy.astype(float), z, z, z, z, z
    if ax is None or ay is None:
        ax, ay = fx, fy
    dt = float(np.median(np.diff(t))) or 1e-3
    want = int(round(window_sec / dt)) | 1
    w = odd_window(n, max(5, want))

    def sm(a: np.ndarray) -> np.ndarray:
        return savgol_filter(a, w, min(order, w - 1)) if w else a.astype(float)

    sx, sy = sm(fx), sm(fy)
    s = scale(sy)
    vx = np.gradient(sm(ax), t) / s
    vy = np.gradient(sm(ay), t) / s
    speed = np.hypot(vx, vy)
    heading = np.degrees(np.arctan2(vy, vx))
    sp_s = savgol_filter(speed, w, min(order, w - 1)) if w else speed
    accel = np.gradient(sp_s, t)
    return sx, sy, vx, vy, speed, heading, accel


def _ls_slope(t: np.ndarray, v: np.ndarray) -> float:
    tc = t - t.mean()
    den = float((tc * tc).sum())
    return float((tc * (v - v.mean())).sum() / den) if den > 1e-9 else 0.0


class _OnlineState:
    __slots__ = ("t", "x", "y", "vx", "vy", "hist", "pos", "cls", "box", "first_t", "n", "edge", "conf")

    def __init__(self, t: float, x: float, y: float, cls: int, box: np.ndarray, maxlen: int) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.cls = cls
        self.box = box
        self.first_t = t
        self.n = 1
        self.edge = False
        self.conf = 0.0  # running mean detection confidence
        self.pos: deque = deque(maxlen=maxlen)  # (t, anchor_x, anchor_y, ground_y)
        self.hist: deque = deque(maxlen=maxlen)  # (t, speed, heading, vx, vy)

    def age(self) -> float:
        return self.t - self.first_t


class OnlineKinematics:
    """Causal kinematics for Part B.

    Velocity is the least-squares slope of the box-top anchor over the last ``vel_window_sec``
    (robust to bottom occlusion); ground position is the box top plus the median box height.
    Only past observations influence the state.
    """

    def __init__(
        self,
        vel_window_sec: float = 0.8,
        history_sec: float = 3.0,
        rate_hz: float = 10.0,
        drop_after_sec: float = 2.0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self.vel_window = vel_window_sec
        self.maxlen = max(6, int(history_sec * rate_hz) + 2)
        self.drop_after = drop_after_sec
        self.width = width
        self.height = height
        self.states: dict[int, _OnlineState] = {}

    def reset(self) -> None:
        self.states.clear()

    def update(self, t: float, tracks: np.ndarray, scale: RowScale) -> None:
        """tracks (M, 7): x1,y1,x2,y2,track_id,conf,cls."""
        seen = set()
        for row in tracks:
            tid = int(row[4])
            seen.add(tid)
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            ax = (x1 + x2) / 2.0
            st = self.states.get(tid)
            if st is None:
                st = _OnlineState(t, ax, y2, int(row[6]), row[:4].copy(), self.maxlen)
                self.states[tid] = st
            st.pos.append((t, ax, y1, y2 - y1))
            st.conf += (float(row[5]) - st.conf) / len(st.pos) if len(st.pos) < st.pos.maxlen else 0.1 * (float(row[5]) - st.conf)
            st.box = row[:4].copy()
            st.cls = int(row[6])
            st.t = t
            st.n += 1
            mx, my = max(2.0, 0.005 * self.width), max(2.0, 0.005 * self.height)
            st.edge = (x1 <= mx) or (y1 <= my) or bool(self.width and x2 >= self.width - mx) or bool(self.height and y2 >= self.height - my)
            p = np.array([q for q in st.pos if q[0] >= t - self.vel_window])
            h_med = float(np.median([q[3] for q in st.pos]))
            st.x = ax
            st.y = y1 + h_med
            if len(p) >= 3:
                s = float(scale(st.y))
                st.vx = _ls_slope(p[:, 0], p[:, 1]) / s
                st.vy = _ls_slope(p[:, 0], p[:, 2]) / s
                speed = float(np.hypot(st.vx, st.vy))
                st.hist.append((t, speed, float(np.degrees(np.arctan2(st.vy, st.vx))), st.vx, st.vy))
        for tid in [k for k, s in self.states.items() if k not in seen and t - s.t > self.drop_after]:
            del self.states[tid]

    def active(self, t: float, max_age: float = 0.5) -> list[tuple[int, _OnlineState]]:
        return [(k, s) for k, s in sorted(self.states.items()) if t - s.t <= max_age]
