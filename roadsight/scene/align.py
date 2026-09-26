"""Align a hand-drawn scene to the current video when the camera has moved slightly.

The scene is drawn once on a reference frame (``scene.json`` -> ``reference_image``). For each video,
SIFT features of the reference and of a frame of the video are matched and a homography is fitted
with RANSAC. The homography is expressed in normalised coordinates (0..1 on both images), so it maps
``scene.json`` points straight to the current video. Deterministic: SIFT, a fixed-seed RANSAC and a
fixed working resolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from roadsight.utils.log import get_logger

log = get_logger("roadsight.align")
WORK_WIDTH = 1280


def _prep(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    scale = WORK_WIDTH / float(w)
    small = cv2.resize(img, (WORK_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    # CLAHE keeps road markings and facades comparable between noon sun and dusk.
    return cv2.createCLAHE(3.0, (8, 8)).apply(gray), float(small.shape[1]), float(small.shape[0])


def normalized_homography(ref: np.ndarray, cur: np.ndarray, min_inliers: int = 40) -> tuple[np.ndarray | None, dict]:
    """Homography mapping normalised reference coordinates to normalised current-frame coordinates."""
    g1, w1, h1 = _prep(ref)
    g2, w2, h2 = _prep(cur)
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    info = {"matches": 0, "inliers": 0}
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, info
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    info["matches"] = len(good)
    if len(good) < min_inliers:
        return None, info
    src = np.float32([k1[m.queryIdx].pt for m in good]) / np.float32([w1, h1])
    dst = np.float32([k2[m.trainIdx].pt for m in good]) / np.float32([w2, h2])
    cv2.setRNGSeed(0)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0 / WORK_WIDTH, maxIters=5000, confidence=0.999)
    inliers = int(mask.sum()) if mask is not None else 0
    info["inliers"] = inliers
    if H is None or inliers < min_inliers or not _plausible(H):
        return None, info
    return H, info


def _plausible(H: np.ndarray) -> bool:
    """Reject wild fits: the unit square must stay convex and roughly the same size."""
    corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area = cv2.contourArea(warped.astype(np.float32))
    return bool(0.5 < area < 2.0 and cv2.isContourConvex(warped.astype(np.float32)))


def warp_points(points, H: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if H is None or len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
