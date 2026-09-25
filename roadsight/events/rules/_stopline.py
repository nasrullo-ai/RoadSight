"""Shared helpers for stop-line rules (red_light, stop_line)."""

from __future__ import annotations

import numpy as np

from roadsight.scene.geometry import path_crosses_polyline, side_of_polyline


def front_points(g, scale) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle front proxy: foot point pushed half a body length along the direction of travel."""
    x = g["sx"].to_numpy()
    y = g["sy"].to_numpy()
    sp = np.maximum(g["speed"].to_numpy(), 1e-9)
    s = scale(y)
    moving = sp > 0.3
    ux = np.where(moving, g["vx"].to_numpy() / sp, 0.0)
    uy = np.where(moving, g["vy"].to_numpy() / sp, 0.0)
    return x + 0.5 * s * ux, y + 0.5 * s * uy


def approach_side(line, vehicle_groups, scale) -> float:
    """Side (+1/-1) vehicles come from: from the configured approach vector, else the majority of crossings."""
    pts = line.points
    if line.approach is not None:
        mid = pts.mean(0) - line.approach * 10.0
        return float(side_of_polyline(np.array([mid[0]]), np.array([mid[1]]), pts)[0])
    votes = 0.0
    for g in vehicle_groups:
        fx, fy = front_points(g, scale)
        if len(fx) < 2:
            continue
        p = np.stack([fx, fy], 1)
        cross = path_crosses_polyline(p[:-1], p[1:], pts)
        if cross.any():
            k = int(np.flatnonzero(cross)[0])
            votes += side_of_polyline(fx[k : k + 1], fy[k : k + 1], pts)[0]
    return 1.0 if votes >= 0 else -1.0


def crossing_index(g, line, side_from: float, scale) -> int | None:
    """Index of the first sample after the front crosses ``line`` from ``side_from``."""
    fx, fy = front_points(g, scale)
    if len(fx) < 2:
        return None
    p = np.stack([fx, fy], 1)
    cross = path_crosses_polyline(p[:-1], p[1:], line.points)
    for k in np.flatnonzero(cross):
        if side_of_polyline(fx[k : k + 1], fy[k : k + 1], line.points)[0] == side_from:
            return int(k + 1)
    return None
