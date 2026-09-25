"""Scene elements learned from tracks when ``scene.json`` does not provide them.

* road mask      -- cells that several distinct vehicle tracks drove through
* flow field     -- per-cell dominant direction of moving vehicles (one vote per track)
* crosswalk mask -- road cells that many distinct pedestrians walked across

Everything lives on a coarse grid (about 64 cells across the frame).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

VEHICLE_CLASSES = (2, 3, 5, 7)
PERSON_CLASS = 0


class Grid:
    def __init__(self, width: int, height: int, cols: int = 64) -> None:
        self.width = width
        self.height = height
        self.cell = max(4.0, width / float(cols))
        self.gw = int(np.ceil(width / self.cell))
        self.gh = int(np.ceil(height / self.cell))

    def index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip((np.asarray(x, dtype=np.float64) / self.cell).astype(int), 0, self.gw - 1)
        iy = np.clip((np.asarray(y, dtype=np.float64) / self.cell).astype(int), 0, self.gh - 1)
        return iy, ix


class FlowField:
    """Dominant travel direction per grid cell.

    In batch mode each track casts one unit vote per visited cell (``add_track``), which allows
    leave-one-out queries so a wrong-way vehicle does not vote for its own direction. In online
    mode (Part B) observations are accumulated with ``add_samples``.
    """

    def __init__(self, grid: Grid, min_support: float = 3.0, blur: int = 1) -> None:
        self.grid = grid
        self.min_support = min_support
        self.blur = blur
        self.sum = np.zeros((grid.gh, grid.gw, 2))
        self.n = np.zeros((grid.gh, grid.gw))
        self._contrib: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._smooth: tuple[np.ndarray, np.ndarray] | None = None
        self._contrib_smooth: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _blur(self, a: np.ndarray) -> np.ndarray:
        if self.blur <= 0:
            return a
        k = 2 * self.blur + 1
        if a.ndim == 3:
            return np.stack([ndimage.uniform_filter(a[..., i], size=k, mode="constant") * k * k for i in range(a.shape[2])], -1)
        return ndimage.uniform_filter(a, size=k, mode="constant") * k * k

    def add_track(self, track_id: int, x: np.ndarray, y: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> None:
        if len(x) == 0:
            return
        iy, ix = self.grid.index(x, y)
        flat = iy * self.grid.gw + ix
        cells, inv = np.unique(flat, return_inverse=True)
        vx = np.bincount(inv, weights=ux, minlength=len(cells))
        vy = np.bincount(inv, weights=uy, minlength=len(cells))
        norm = np.hypot(vx, vy)
        ok = norm > 1e-6
        cells, vx, vy, norm = cells[ok], vx[ok], vy[ok], norm[ok]
        vec = np.stack([vx / norm, vy / norm], -1)
        self.sum.reshape(-1, 2)[cells] += vec
        self.n.reshape(-1)[cells] += 1.0
        self._contrib[track_id] = (cells, vec)
        self._smooth = None

    def add_samples(self, x: np.ndarray, y: np.ndarray, ux: np.ndarray, uy: np.ndarray, weight: float = 1.0) -> None:
        if len(x) == 0:
            return
        iy, ix = self.grid.index(x, y)
        np.add.at(self.sum, (iy, ix), np.stack([ux, uy], -1) * weight)
        np.add.at(self.n, (iy, ix), weight)
        self._smooth = None

    def _smoothed(self) -> tuple[np.ndarray, np.ndarray]:
        if self._smooth is None:
            self._smooth = (self._blur(self.sum), self._blur(self.n))
        return self._smooth

    def _track_smoothed(self, track_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if track_id not in self._contrib:
            return None
        if track_id not in self._contrib_smooth:
            cells, vec = self._contrib[track_id]
            s = np.zeros_like(self.sum)
            n = np.zeros_like(self.n)
            s.reshape(-1, 2)[cells] = vec
            n.reshape(-1)[cells] = 1.0
            self._contrib_smooth[track_id] = (self._blur(s), self._blur(n))
        return self._contrib_smooth[track_id]

    def direction(self, x, y, exclude_track: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        s_all, n_all = self._smoothed()
        iy, ix = self.grid.index(x, y)
        s = s_all[iy, ix].copy()
        n = n_all[iy, ix].copy()
        if exclude_track is not None:
            own = self._track_smoothed(exclude_track)
            if own is not None:
                s -= own[0][iy, ix]
                n -= own[1][iy, ix]
        mean = s / np.maximum(n, 1e-9)[:, None]
        coherence = np.hypot(mean[:, 0], mean[:, 1])
        dirs = mean / np.maximum(coherence, 1e-9)[:, None]
        strength = np.where(n >= self.min_support, np.clip(coherence, 0, 1), 0.0)
        return dirs, strength


class AutoScene:
    def __init__(self, grid: Grid, road: np.ndarray, flow: FlowField, crosswalk: np.ndarray | None) -> None:
        self.grid = grid
        self.road = road
        self.flow = flow
        self.crosswalk = crosswalk
        depth = cv2.distanceTransform(np.pad(road.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        self.road_depth = depth * grid.cell

    def lookup(self, grid_arr: np.ndarray, x, y) -> np.ndarray:
        iy, ix = self.grid.index(x, y)
        return grid_arr[iy, ix]

    @classmethod
    def build(cls, df, width: int, height: int, cfg: dict) -> AutoScene:
        """Build from a TrackTable DataFrame with columns track_id, cls, foot_x, foot_y, vx, vy, speed, rider."""
        grid = Grid(width, height, int(cfg.get("grid_cols", 64)))
        road_counts = np.zeros((grid.gh, grid.gw))
        flow = FlowField(grid, min_support=float(cfg.get("flow_min_tracks", 3)), blur=int(cfg.get("flow_blur", 1)))
        v_move = float(cfg.get("moving_speed", 0.5))

        veh = df[df["cls"].isin(VEHICLE_CLASSES)]
        for tid, g in veh.groupby("track_id", sort=True):
            iy, ix = grid.index(g["foot_x"].to_numpy(), g["foot_y"].to_numpy())
            cells = np.unique(iy * grid.gw + ix)
            road_counts.reshape(-1)[cells] += 1
            mv = g[g["speed"] > v_move]
            if len(mv):
                sp = np.hypot(mv["vx"].to_numpy(), mv["vy"].to_numpy())
                sp = np.maximum(sp, 1e-9)
                flow.add_track(
                    int(tid), mv["foot_x"].to_numpy(), mv["foot_y"].to_numpy(), mv["vx"].to_numpy() / sp, mv["vy"].to_numpy() / sp
                )

        n_tracks = veh["track_id"].nunique() if len(veh) else 0
        min_tracks = max(float(cfg.get("road_min_tracks", 2)), float(cfg.get("road_min_frac", 0.0)) * n_tracks)
        road = road_counts >= min_tracks
        if road.any():
            st = np.ones((3, 3), dtype=bool)
            road = ndimage.binary_closing(road, structure=st, iterations=int(cfg.get("road_close", 2)))
            road = ndimage.binary_dilation(road, structure=st, iterations=int(cfg.get("road_dilate", 1)))
            road = ndimage.binary_fill_holes(road)

        crosswalk = None
        if cfg.get("auto_crosswalks", True):
            ped = df[(df["cls"] == PERSON_CLASS) & (~df["rider"].astype(bool)) & (df["speed"] > 0.2)]
            counts = np.zeros((grid.gh, grid.gw))
            for _, g in ped.groupby("track_id", sort=True):
                iy, ix = grid.index(g["foot_x"].to_numpy(), g["foot_y"].to_numpy())
                counts.reshape(-1)[np.unique(iy * grid.gw + ix)] += 1
            cw = (counts >= float(cfg.get("crosswalk_min_persons", 4))) & road
            if cw.any():
                cw = ndimage.binary_dilation(cw, structure=np.ones((3, 3), dtype=bool), iterations=1)
                crosswalk = cw
        return cls(grid, road, flow, crosswalk)

    @classmethod
    def empty(cls, width: int, height: int, cfg: dict) -> AutoScene:
        grid = Grid(width, height, int(cfg.get("grid_cols", 64)))
        flow = FlowField(grid, min_support=float(cfg.get("online_flow_min_support", 30)), blur=int(cfg.get("flow_blur", 1)))
        return cls(grid, np.zeros((grid.gh, grid.gw), dtype=bool), flow, None)
