"""Scene configuration: hand-drawn geometry from ``scene.json`` plus automatic fall-backs.

``scene.json`` stores coordinates normalised to [0, 1] (``"normalized": true``) so it survives
resolution changes. Every element is optional; rules that need a missing element disable
themselves. Carriageway, lane flow and crosswalks fall back to an :class:`AutoScene` learned
from the tracks of the video itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from roadsight.config import resolve_path
from roadsight.scene import geometry as geo

if TYPE_CHECKING:
    from roadsight.scene.auto import AutoScene


@dataclass
class Lane:
    id: str
    polygon: np.ndarray
    direction: np.ndarray  # unit vector in image coordinates


@dataclass
class StopLine:
    id: str
    points: np.ndarray
    signal: str | None
    approach: np.ndarray | None  # unit vector of travel when crossing, or None to infer


@dataclass
class Zone:
    id: str
    polygon: np.ndarray


@dataclass
class SignalROI:
    id: str
    rect: tuple[int, int, int, int]


@dataclass
class Scene:
    width: int
    height: int
    carriageway: list[np.ndarray] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    crosswalks: list[np.ndarray] = field(default_factory=list)
    solid_lines: list[np.ndarray] = field(default_factory=list)
    no_u_turn_zones: list[np.ndarray] = field(default_factory=list)
    exit_zones: list[Zone] = field(default_factory=list)
    allowed_movements: set[tuple[str, str]] = field(default_factory=set)
    signal_rois: list[SignalROI] = field(default_factory=list)
    queue_zones: list[np.ndarray] = field(default_factory=list)
    intersection: list[np.ndarray] = field(default_factory=list)
    fixed_objects: list[np.ndarray] = field(default_factory=list)
    right_turn_on_red: bool = False
    auto: AutoScene | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None, width: int, height: int) -> Scene:
        """Load scene.json scaled to a ``width`` x ``height`` frame; missing file -> empty scene."""
        scene = cls(width=width, height=height)
        if not path:
            return scene
        p = resolve_path(path)
        if not p.exists():
            return scene
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("normalized", True):
            sx, sy = float(width), float(height)
        else:
            iw, ih = data.get("image_size", [width, height])
            sx, sy = width / float(iw), height / float(ih)

        def pts(v) -> np.ndarray:
            a = geo.as_points(v)
            return a * np.array([sx, sy]) if len(a) else a

        def unit(v) -> np.ndarray:
            d = np.asarray(v, dtype=np.float64) * np.array([sx, sy])
            n = np.linalg.norm(d)
            return d / n if n > 0 else d

        scene.carriageway = [pts(p) for p in data.get("carriageway", [])]
        scene.lanes = [Lane(str(ln.get("id", i)), pts(ln["polygon"]), unit(ln["direction"])) for i, ln in enumerate(data.get("lanes", []))]
        scene.stop_lines = [
            StopLine(str(s.get("id", i)), pts(s["points"]), s.get("signal"), unit(s["approach"]) if s.get("approach") else None)
            for i, s in enumerate(data.get("stop_lines", []))
        ]
        scene.crosswalks = [pts(p) for p in data.get("crosswalks", [])]
        scene.solid_lines = [pts(p) for p in data.get("solid_lines", [])]
        scene.no_u_turn_zones = [pts(p) for p in data.get("no_u_turn_zones", [])]
        scene.exit_zones = [Zone(str(z.get("id", i)), pts(z["polygon"])) for i, z in enumerate(data.get("exit_zones", []))]
        scene.allowed_movements = {(str(a), str(b)) for a, b in data.get("allowed_movements", [])}
        scene.signal_rois = []
        for i, r in enumerate(data.get("signal_rois", [])):
            x1, y1, x2, y2 = r["rect"]
            scene.signal_rois.append(SignalROI(str(r.get("id", i)), (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))))
        scene.queue_zones = [pts(p) for p in data.get("queue_zones", [])]
        scene.intersection = [pts(p) for p in data.get("intersection", [])]
        scene.fixed_objects = [pts(p) for p in data.get("fixed_objects", [])]
        scene.right_turn_on_red = bool(data.get("right_turn_on_red", False))
        return scene

    # ------------------------------------------------------------------ queries
    def has_carriageway(self) -> bool:
        return bool(self.carriageway) or (self.auto is not None and self.auto.road.any())

    def in_carriageway(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.carriageway:
            return geo.points_in_any(x, y, self.carriageway)
        if self.auto is not None:
            return self.auto.lookup(self.auto.road, x, y)
        return np.zeros(x.shape, dtype=bool)

    def depth_in_carriageway(self, x, y) -> np.ndarray:
        """Distance (px) from each point to the carriageway edge; 0 outside."""
        if self.auto is not None and self.auto.road_depth is not None and not self.carriageway:
            return self.auto.lookup(self.auto.road_depth, x, y).astype(np.float64)
        if self.carriageway:
            if not hasattr(self, "_cw_depth"):
                mask = geo.rasterize(self.carriageway, self.width, self.height, cell=4).astype(np.uint8)
                self._cw_depth = cv2.distanceTransform(mask, cv2.DIST_L2, 3) * 4.0
            ix = np.clip((np.asarray(x) / 4).astype(int), 0, self._cw_depth.shape[1] - 1)
            iy = np.clip((np.asarray(y) / 4).astype(int), 0, self._cw_depth.shape[0] - 1)
            return self._cw_depth[iy, ix]
        return np.zeros(np.shape(x))

    def in_crosswalk(self, x, y, buffer_px: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.crosswalks:
            inside = geo.points_in_any(x, y, self.crosswalks)
            if buffer_px > 0:
                for cw in self.crosswalks:
                    closed = np.vstack([cw, cw[:1]])
                    inside |= geo.distance_to_polyline(x, y, closed) <= buffer_px
            return inside
        if self.auto is not None and self.auto.crosswalk is not None:
            return self.auto.lookup(self.auto.crosswalk, x, y)
        return np.zeros(x.shape, dtype=bool)

    def has_crosswalks(self) -> bool:
        return bool(self.crosswalks) or (self.auto is not None and self.auto.crosswalk is not None and self.auto.crosswalk.any())

    def lane_direction(self, x, y, exclude_track: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Expected travel direction at each point: (unit vectors (N,2), strength (N,)).

        Strength is 1 for hand-drawn lanes, the flow-field coherence (0..1) for automatic lanes,
        and 0 where no direction is known.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        dirs = np.zeros((len(x), 2))
        strength = np.zeros(len(x))
        if self.lanes:
            for lane in self.lanes:
                m = geo.points_in_polygon(x, y, lane.polygon) & (strength == 0)
                dirs[m] = lane.direction
                strength[m] = 1.0
            return dirs, strength
        if self.auto is not None:
            return self.auto.flow.direction(x, y, exclude_track)
        return dirs, strength

    def lane_id(self, x, y) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(len(x), "", dtype=object)
        for lane in self.lanes:
            m = geo.points_in_polygon(x, y, lane.polygon) & (out == "")
            out[m] = lane.id
        return out

    def zone_of(self, x: float, y: float, zones: list[Zone]) -> str | None:
        for z in zones:
            if geo.points_in_polygon(np.array([x]), np.array([y]), z.polygon)[0]:
                return z.id
        return None

    def to_overlay(self) -> dict:
        """Pixel geometry for rendering (lists, JSON-friendly)."""

        def lst(polys):
            return [np.asarray(p).round(1).tolist() for p in polys]

        return {
            "carriageway": lst(self.carriageway),
            "lanes": [
                {"id": ln.id, "polygon": ln.polygon.round(1).tolist(), "direction": ln.direction.round(3).tolist()} for ln in self.lanes
            ],
            "stop_lines": [{"id": s.id, "points": s.points.round(1).tolist()} for s in self.stop_lines],
            "crosswalks": lst(self.crosswalks),
            "solid_lines": lst(self.solid_lines),
            "no_u_turn_zones": lst(self.no_u_turn_zones),
            "exit_zones": [{"id": z.id, "polygon": z.polygon.round(1).tolist()} for z in self.exit_zones],
            "queue_zones": lst(self.queue_zones),
            "signal_rois": [{"id": r.id, "rect": list(r.rect)} for r in self.signal_rois],
        }
