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
from roadsight.scene.align import warp_points

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
    kind: str = "vehicle"  # vehicle | pedestrian


@dataclass
class Scene:
    width: int
    height: int
    carriageway: list[np.ndarray] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    crosswalks: list[np.ndarray] = field(default_factory=list)
    crosswalk_signals: list[str | None] = field(default_factory=list)  # vehicle signal governing each crosswalk
    non_road: list[np.ndarray] = field(default_factory=list)  # islands, medians, sidewalks inside the carriageway
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
    @staticmethod
    def read_json(path: str | Path | None) -> dict | None:
        if not path:
            return None
        p = resolve_path(path)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def reference_image(path: str | Path | None) -> Path | None:
        """Reference frame the scene was drawn on (for alignment), if the scene names one that exists."""
        data = Scene.read_json(path)
        ref = (data or {}).get("reference_image")
        if not ref or not resolve_path(ref).exists():
            return None
        return resolve_path(ref)

    @classmethod
    def load(cls, path: str | Path | None, width: int, height: int, homography: np.ndarray | None = None, geometry: bool = True) -> Scene:
        """Load scene.json scaled to a ``width`` x ``height`` frame; missing file -> empty scene.

        ``homography`` (normalised reference -> normalised current frame, see ``scene.align``) moves the
        drawn geometry onto a slightly different camera view. ``geometry=False`` keeps only the flags
        (used when the scene could not be aligned to this video).
        """
        scene = cls(width=width, height=height)
        data = cls.read_json(path)
        if data is None:
            return scene
        scene.right_turn_on_red = bool(data.get("right_turn_on_red", False))
        if not geometry:
            return scene
        if data.get("normalized", True):
            nx, ny = 1.0, 1.0
        else:
            iw, ih = data.get("image_size", [width, height])
            nx, ny = 1.0 / float(iw), 1.0 / float(ih)
        size = np.array([float(width), float(height)])

        def norm(v) -> np.ndarray:
            a = geo.as_points(v)
            if not len(a):
                return a
            return warp_points(a * np.array([nx, ny]), homography)

        def pts(v) -> np.ndarray:
            a = norm(v)
            return a * size if len(a) else a

        def unit(v, anchor) -> np.ndarray:
            """Direction ``v`` (drawn at ``anchor``) after the warp, in pixel space."""
            c = np.asarray(anchor, dtype=np.float64).reshape(2) * np.array([nx, ny])
            d = np.asarray(v, dtype=np.float64).reshape(2) * np.array([nx, ny])
            d = d / max(np.linalg.norm(d), 1e-9) * 0.02
            a, b = warp_points(np.vstack([c, c + d]), homography) * size
            u = b - a
            n = np.linalg.norm(u)
            return u / n if n > 0 else u

        scene.carriageway = [pts(p) for p in data.get("carriageway", [])]
        scene.lanes = [
            Lane(str(ln.get("id", i)), pts(ln["polygon"]), unit(ln["direction"], geo.as_points(ln["polygon"]).mean(0)))
            for i, ln in enumerate(data.get("lanes", []))
        ]
        scene.stop_lines = [
            StopLine(
                str(s.get("id", i)),
                pts(s["points"]),
                s.get("signal"),
                unit(s["approach"], geo.as_points(s["points"]).mean(0)) if s.get("approach") else None,
            )
            for i, s in enumerate(data.get("stop_lines", []))
        ]
        cws = data.get("crosswalks", [])
        scene.crosswalks = [pts(c["polygon"] if isinstance(c, dict) else c) for c in cws]
        scene.crosswalk_signals = [c.get("signal") if isinstance(c, dict) else None for c in cws]
        scene.non_road = [pts(p) for p in data.get("non_road", [])]
        scene.solid_lines = [pts(p) for p in data.get("solid_lines", [])]
        scene.no_u_turn_zones = [pts(p) for p in data.get("no_u_turn_zones", [])]
        scene.exit_zones = [Zone(str(z.get("id", i)), pts(z["polygon"])) for i, z in enumerate(data.get("exit_zones", []))]
        scene.allowed_movements = {(str(a), str(b)) for a, b in data.get("allowed_movements", [])}
        scene.signal_rois = []
        for i, r in enumerate(data.get("signal_rois", [])):
            x1, y1, x2, y2 = r["rect"]
            c = pts([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            rect = (int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 0].max()), int(c[:, 1].max()))
            scene.signal_rois.append(SignalROI(str(r.get("id", i)), rect, str(r.get("kind", "vehicle"))))
        scene.queue_zones = [pts(p) for p in data.get("queue_zones", [])]
        scene.intersection = [pts(p) for p in data.get("intersection", [])]
        scene.fixed_objects = [pts(p) for p in data.get("fixed_objects", [])]
        return scene

    # ------------------------------------------------------------------ queries
    def has_carriageway(self) -> bool:
        return bool(self.carriageway) or (self.auto is not None and self.auto.road.any())

    def in_carriageway(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.carriageway:
            inside = geo.points_in_any(x, y, self.carriageway)
            if self.non_road:
                inside &= ~geo.points_in_any(x, y, self.non_road)
            return inside
        if self.auto is not None:
            inside = self.auto.lookup(self.auto.road, x, y)
            if self.non_road:
                inside = inside & ~geo.points_in_any(x, y, self.non_road)
            return inside
        return np.zeros(x.shape, dtype=bool)

    def depth_in_carriageway(self, x, y) -> np.ndarray:
        """Distance (px) from each point to the carriageway edge; 0 outside."""
        if self.auto is not None and self.auto.road_depth is not None and not self.carriageway:
            return self.auto.lookup(self.auto.road_depth, x, y).astype(np.float64)
        if self.carriageway:
            if not hasattr(self, "_cw_depth"):
                mask = geo.rasterize(self.carriageway, self.width, self.height, cell=4)
                if self.non_road:
                    mask &= ~geo.rasterize(self.non_road, self.width, self.height, cell=4)
                mask = mask.astype(np.uint8)
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

    def vehicle_signal(self) -> str | None:
        """Id of the first vehicle signal head, the default for rules that need one phase."""
        return next((r.id for r in self.signal_rois if r.kind == "vehicle"), None)

    def crosswalk_index(self, x, y) -> np.ndarray:
        """Index of the drawn crosswalk containing each point, -1 when none."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        out = np.full(len(x), -1)
        for i, cw in enumerate(self.crosswalks):
            out[(out < 0) & geo.points_in_polygon(x, y, cw)] = i
        return out

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


_REF_CACHE: dict[str, np.ndarray] = {}


def load_aligned(path: str | Path | None, frame: np.ndarray | None, width: int, height: int, min_inliers: int = 40) -> tuple[Scene, dict]:
    """Load the scene and, when it names a reference image, align it to ``frame``.

    If alignment fails the drawn geometry is dropped (automatic scene only): misplaced stop lines or
    crosswalks would create false events, which is worse than having none.
    """
    ref_path = Scene.reference_image(path)
    if ref_path is None or frame is None:
        return Scene.load(path, width, height), {"aligned": None}
    from roadsight.scene.align import normalized_homography
    from roadsight.utils.log import get_logger

    key = str(ref_path)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = cv2.imread(key)
    H, info = normalized_homography(_REF_CACHE[key], frame, min_inliers)
    if H is None:
        get_logger("roadsight.scene").warning("scene alignment failed (%s); using the automatic scene only", info)
        return Scene.load(path, width, height, geometry=False), {"aligned": False, **info}
    return Scene.load(path, width, height, homography=H), {"aligned": True, **info, "H": H.round(5).tolist()}


def first_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
