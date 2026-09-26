"""Scene alignment: a shifted, relit copy of the reference frame maps the drawn geometry correctly."""

import json

import cv2
import numpy as np
import pytest

from roadsight.config import resolve_path
from roadsight.scene.align import normalized_homography, warp_points
from roadsight.scene.scene import Scene, load_aligned

REF = resolve_path("configs/scene_ref.jpg")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="no reference image")


def shifted(img: np.ndarray, dx: float, dy: float, zoom: float, gain: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    M = np.array([[zoom, 0, dx + (1 - zoom) * w / 2], [0, zoom, dy + (1 - zoom) * h / 2]], dtype=np.float64)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8), M


def test_alignment_recovers_camera_shift():
    ref = cv2.imread(str(REF))
    cur, M = shifted(ref, dx=40, dy=-25, zoom=1.06, gain=0.55)  # moved, zoomed and darker (dusk)
    H, info = normalized_homography(ref, cur)
    assert H is not None, info
    h, w = ref.shape[:2]
    probe = np.array([[0.2, 0.5], [0.6, 0.45], [0.35, 0.9]])
    got = warp_points(probe, H) * [w, h]
    want = np.c_[probe * [w, h], np.ones(3)] @ M.T
    assert np.abs(got - want).max() < 4.0


def test_scene_follows_the_camera(tmp_path):
    ref = cv2.imread(str(REF))
    cur, M = shifted(ref, dx=-30, dy=20, zoom=1.0, gain=1.0)
    scene, info = load_aligned("configs/scene.json", cur, ref.shape[1], ref.shape[0])
    assert info["aligned"]
    base = Scene.load("configs/scene.json", ref.shape[1], ref.shape[0])
    moved = np.c_[base.stop_lines[0].points, np.ones(2)] @ M.T
    assert np.abs(scene.stop_lines[0].points - moved).max() < 4.0


def test_unrelated_frame_drops_drawn_geometry():
    noise = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    scene, info = load_aligned("configs/scene.json", noise, 1920, 1080)
    assert info["aligned"] is False
    assert not scene.stop_lines and not scene.crosswalks and not scene.signal_rois
    data = json.loads(resolve_path("configs/scene.json").read_text(encoding="utf-8"))
    assert data["stop_lines"], "the drawn scene itself is intact"
