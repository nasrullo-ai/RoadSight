"""Causal danger measurements for Part B (SPEC section 10). Inputs are online track states only.

``measure`` returns raw quantities (seconds, BL/s^2, deg/s, flags); the model turns them into
0..1 features, normalising braking and swerving by the video's own recent noise level.
"""

from __future__ import annotations

import numpy as np

from roadsight.perception.interactions import candidate_pairs, pair_ttc
from roadsight.perception.kinematics import _ls_slope
from roadsight.scene.geometry import angle_diff_deg, side_of_polyline

FEATURES = ("f_ttc", "f_brake", "f_swerve", "f_conflict", "f_wrong", "f_redrun", "f_ped", "f_density")
VEHICLES = (2, 3, 5, 7)


def measure(active, t: float, scale, scene, red_lines: list, cfg: dict, height: int, pair_hist: dict | None = None) -> dict[str, float]:
    """``active``: list of (track_id, OnlineState). ``red_lines``: (stop line, approach side) pairs that are red now.

    ``pair_hist`` (owned by the caller, causal) keeps each pair's recent distances so only pairs whose gap has
    really been shrinking over the last second count for time to collision.
    """
    raw = {"ttc": np.inf, "ped_ttc": np.inf, "decel": 0.0, "swerve": 0.0, "wrong": 0.0, "redrun": 0.0, "ped": 0.0, "n_veh": 0.0}
    if not active:
        return raw
    min_size = float(cfg.get("min_size_frac", 0.03)) * height
    min_age = float(cfg.get("min_age_sec", 1.0))
    min_conf = float(cfg.get("min_track_conf", 0.4))
    keep = []
    for k, s in active:
        w, h = s.box[2] - s.box[0], s.box[3] - s.box[1]
        if np.sqrt(max(w, 1) * max(h, 1)) >= min_size and not s.edge and s.age() >= min_age and len(s.hist) >= 3 and s.conf >= min_conf:
            keep.append((k, s))
    raw["n_veh"] = float(sum(1 for _, s in active if s.cls in VEHICLES and s.conf >= min_conf))
    if not keep:
        return raw
    ids = np.array([k for k, _ in keep])
    cls = np.array([s.cls for _, s in keep])
    x = np.array([s.x for _, s in keep])
    y = np.array([s.y for _, s in keep])
    vx = np.array([s.vx for _, s in keep])
    vy = np.array([s.vy for _, s in keep])
    boxes = np.array([s.box for _, s in keep], dtype=np.float64)
    sp = np.hypot(vx, vy)
    sc = scale(y)
    is_veh = np.isin(cls, VEHICLES)
    is_ped = cls == 0

    # Time to collision over nearby, closing pairs.
    pairs = candidate_pairs(x, y, float(cfg.get("pair_dist_bl", 6.0)) * sc)
    if len(pairs):
        pairs = pairs[((sp[pairs[:, 0]] > 0.5) | (sp[pairs[:, 1]] > 0.5)) & (is_veh[pairs[:, 0]] | is_veh[pairs[:, 1]])]
    auto = getattr(scene, "auto", None)
    if len(pairs) and auto is not None and getattr(auto, "queue", None) is not None:
        # A vehicle closing on a (nearly) standing one inside a learned queue zone is joining the queue.
        in_q = auto.lookup(auto.queue, x, y)
        slow = np.minimum(sp[pairs[:, 0]], sp[pairs[:, 1]]) < float(cfg.get("queue_leader_speed", 0.5))
        pairs = pairs[~((in_q[pairs[:, 0]] | in_q[pairs[:, 1]]) & slow)]
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        dx, dy = x[j] - x[i], y[j] - y[i]
        dist = np.maximum(np.hypot(dx, dy), 1e-6)
        rvx, rvy = vx[j] * sc[j] - vx[i] * sc[i], vy[j] * sc[j] - vy[i] * sc[i]
        closing = -(rvx * dx + rvy * dy) / dist / ((sc[i] + sc[j]) / 2)
        ok = closing >= float(cfg.get("min_closing_bl", 1.0))
        if pair_hist is not None:
            gap_bl = dist / ((sc[i] + sc[j]) / 2)
            window = float(cfg.get("closing_window_sec", 1.0))
            for n, (a, b) in enumerate(zip(ids[i], ids[j])):
                h = pair_hist.setdefault((int(min(a, b)), int(max(a, b))), [])
                h.append((t, float(gap_bl[n])))
                while h and h[0][0] < t - window:
                    h.pop(0)
                if len(h) >= 3:
                    ht = np.array([q[0] for q in h])
                    hd = np.array([q[1] for q in h])
                    ok[n] &= -_ls_slope(ht, hd) >= float(cfg.get("min_closing_bl", 1.0))
                else:
                    ok[n] = False
        pairs = pairs[ok]
    if len(pairs):
        vel = np.stack([vx * sc, vy * sc], 1)
        ttc, now = pair_ttc(boxes, vel, pairs, float(cfg.get("horizon_sec", 3.0)))
        ttc = np.where(now, np.inf, ttc)
        raw["ttc"] = float(ttc.min())
        with_ped = is_ped[pairs[:, 0]] | is_ped[pairs[:, 1]]
        if with_ped.any():
            raw["ped_ttc"] = float(ttc[with_ped].min())

    # Braking (LS slope of speed) and swerving (heading rate) over the last second.
    for (_, st), veh in zip(keep, is_veh):
        if not veh:
            continue
        h = [r for r in st.hist if r[0] >= t - 1.0]
        if len(h) < 4:
            continue
        ht = np.array([r[0] for r in h])
        hs = np.array([r[1] for r in h])
        if hs[0] >= 0.8:
            raw["decel"] = max(raw["decel"], -_ls_slope(ht, hs))
        mv = hs > 1.0
        if mv.sum() >= 4:
            hh = np.array([r[2] for r in h])[mv]
            span = ht[mv][-1] - ht[mv][0]
            if span > 0.4:
                raw["swerve"] = max(raw["swerve"], float(np.abs(angle_diff_deg(hh[-1], hh[0])) / span))

    # Wrong way against the (online) lane flow.
    mv = is_veh & (sp > 0.8)
    if mv.any():
        dirs, strength = scene.lane_direction(x[mv], y[mv])
        cos = (np.stack([vx[mv], vy[mv]], 1) / sp[mv][:, None] * dirs).sum(1)
        if ((strength >= float(cfg.get("wrong_min_strength", 0.6))) & (cos < -0.5)).any():
            raw["wrong"] = 1.0

    # Red-light runner: approaching a red stop line faster than it can stop.
    a_max = float(cfg.get("stop_decel", 1.0))
    for line, side_from in red_lines:
        sel = np.flatnonzero(is_veh & (sp > 0.5))
        if not len(sel):
            continue
        side = side_of_polyline(x[sel], y[sel], line.points)
        d = np.min(np.hypot(x[sel, None] - line.points[None, :, 0], y[sel, None] - line.points[None, :, 1]), axis=1) / sc[sel]
        if ((side == side_from) & (d < 3.0) & (sp[sel] ** 2 / (2 * a_max) > d)).any():
            raw["redrun"] = 1.0
            break

    # Pedestrian on the carriageway with a vehicle closing in.
    if is_ped.any() and raw["ped_ttc"] < 3.0:
        px, py = x[is_ped], y[is_ped]
        on_road = scene.in_carriageway(px, py)
        if scene.has_crosswalks():  # people on a crosswalk at a signalised junction are expected there
            on_road &= ~scene.in_crosswalk(px, py, buffer_px=float(np.median(sc)) * 0.5)
        if on_road.any():
            raw["ped"] = 1.0
    return raw
