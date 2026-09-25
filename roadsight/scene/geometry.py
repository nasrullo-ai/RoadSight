"""Vectorised 2-D geometry: polygons, polylines, segment tests. Coordinates are pixels."""

from __future__ import annotations

import cv2
import numpy as np


def as_points(pts) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly) -> np.ndarray:
    """Even-odd ray casting; ``x`` and ``y`` are arrays of equal shape."""
    poly = as_points(poly)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_int)
        j = i
    return inside


def points_in_any(x: np.ndarray, y: np.ndarray, polys) -> np.ndarray:
    out = np.zeros(np.shape(x), dtype=bool)
    for poly in polys:
        out |= points_in_polygon(x, y, poly)
    return out


def rasterize(polys, width: int, height: int, cell: int = 1) -> np.ndarray:
    """Binary mask (height//cell, width//cell) of the union of polygons."""
    h = max(1, int(np.ceil(height / cell)))
    w = max(1, int(np.ceil(width / cell)))
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.round(as_points(poly) / cell).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Intersection test for segment arrays p1-p2 (N,2) against one segment q1-q2.

    Points exactly on a line count as the positive side, so a path touching a line is
    counted as crossing it exactly once.
    """
    p1 = np.atleast_2d(np.asarray(p1, dtype=np.float64))
    p2 = np.atleast_2d(np.asarray(p2, dtype=np.float64))

    def orient(a, b, c):
        v = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        return np.where(v >= 0, 1, -1)

    q1 = np.broadcast_to(np.asarray(q1, dtype=np.float64), p1.shape)
    q2 = np.broadcast_to(np.asarray(q2, dtype=np.float64), p1.shape)
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) & (orient(q1, q2, p1) != orient(q1, q2, p2))


def path_crosses_polyline(p1: np.ndarray, p2: np.ndarray, polyline) -> np.ndarray:
    """Whether each step p1->p2 crosses any segment of ``polyline``."""
    line = as_points(polyline)
    out = np.zeros(len(np.atleast_2d(p1)), dtype=bool)
    for a, b in zip(line[:-1], line[1:]):
        out |= segments_intersect(p1, p2, a, b)
    return out


def side_of_polyline(x: np.ndarray, y: np.ndarray, polyline) -> np.ndarray:
    """Signed side (+1/-1) of points relative to the nearest segment of a polyline."""
    line = as_points(polyline)
    pts = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1).reshape(-1, 2)
    best_d = np.full(len(pts), np.inf)
    side = np.zeros(len(pts))
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = max(float(ab @ ab), 1e-9)
        t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.hypot(*(pts - proj).T)
        cross = ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])
        better = d < best_d
        best_d = np.where(better, d, best_d)
        side = np.where(better, np.sign(cross), side)
    side[side == 0] = 1.0
    return side.reshape(np.shape(x))


def distance_to_polyline(x: np.ndarray, y: np.ndarray, polyline) -> np.ndarray:
    line = as_points(polyline)
    pts = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = max(float(ab @ ab), 1e-9)
        t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.hypot(*(pts - proj).T))
    return best.reshape(np.shape(x))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy."""
    a = np.atleast_2d(a).astype(np.float64)
    b = np.atleast_2d(b).astype(np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def footprint(boxes: np.ndarray, frac: float = 0.35) -> np.ndarray:
    """Lower part of each box, a rough ground-plane footprint that limits perspective overlap."""
    boxes = np.atleast_2d(boxes).astype(np.float64).copy()
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 1] = boxes[:, 3] - frac * h
    return boxes


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return d
