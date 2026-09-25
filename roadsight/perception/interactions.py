"""Pairwise road-user interactions: candidate pairs and time-to-collision (TTC).

TTC uses constant-velocity extrapolation of box *footprints* (the lower part of each box),
which keeps perspective overlap of neighbouring lanes from looking like a collision course.
"""

from __future__ import annotations

import numpy as np

from roadsight.scene.geometry import footprint


def candidate_pairs(x: np.ndarray, y: np.ndarray, max_dist_px: np.ndarray) -> np.ndarray:
    """Index pairs (i < j) closer than the larger of their distance limits."""
    n = len(x)
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    i, j = np.triu_indices(n, k=1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    keep = d < np.maximum(max_dist_px[i], max_dist_px[j])
    return np.stack([i[keep], j[keep]], 1)


def pair_ttc(
    boxes: np.ndarray, vel_px: np.ndarray, pairs: np.ndarray, horizon: float = 3.0, step: float = 0.1, frac: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ttc, overlapping_now) per pair. ttc = inf when no overlap within ``horizon``."""
    if len(pairs) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    fp = footprint(boxes, frac)
    a, b = fp[pairs[:, 0]], fp[pairs[:, 1]]
    va, vb = vel_px[pairs[:, 0]], vel_px[pairs[:, 1]]
    now = overlap(a, b)
    taus = np.arange(step, horizon + 1e-9, step)[None, :]
    dvx = (va[:, 0] - vb[:, 0])[:, None] * taus
    dvy = (va[:, 1] - vb[:, 1])[:, None] * taus
    # Move a relative to b.
    ox = np.minimum(a[:, 2, None] + dvx, b[:, 2, None]) - np.maximum(a[:, 0, None] + dvx, b[:, 0, None])
    oy = np.minimum(a[:, 3, None] + dvy, b[:, 3, None]) - np.maximum(a[:, 1, None] + dvy, b[:, 1, None])
    hit = (ox > 0) & (oy > 0)
    first = np.argmax(hit, axis=1)
    ttc = np.where(hit.any(1), taus[0, first], np.inf)
    ttc = np.where(now, 0.0, ttc)
    return ttc, now


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.minimum(a[:, 2], b[:, 2]) > np.maximum(a[:, 0], b[:, 0])) & (np.minimum(a[:, 3], b[:, 3]) > np.maximum(a[:, 1], b[:, 1]))


def window_mean(t: np.ndarray, v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Mean of ``v`` over samples with lo <= t < hi (per query). NaN where empty."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    i0 = np.searchsorted(t, lo, side="left")
    i1 = np.searchsorted(t, hi, side="left")
    n = i1 - i0
    return np.where(n > 0, (c[i1] - c[i0]) / np.maximum(n, 1), np.nan)
