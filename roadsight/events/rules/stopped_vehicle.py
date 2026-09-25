"""stopped_vehicle: a vehicle stationary for >= 10 s on the carriageway, outside a signal queue."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.scene.geometry import box_iou, points_in_any


def stationary_episodes(t: np.ndarray, speed: np.ndarray, v_stop: float, v_go: float, go_sec: float) -> list[tuple[int, int, bool]]:
    """Hysteresis: start below ``v_stop``, end once above ``v_go`` for ``go_sec``.

    Returns (first_idx, last_stationary_idx, ended_by_track_loss).
    """
    out = []
    n = len(t)
    i = 0
    while i < n:
        if speed[i] >= v_stop:
            i += 1
            continue
        start = i
        j = i
        last_still = i
        go_since = None
        ended_by_loss = True
        while j < n:
            if speed[j] > v_go:
                go_since = t[j] if go_since is None else go_since
                if t[j] - go_since >= go_sec:
                    ended_by_loss = False
                    break
            else:
                go_since = None
                last_still = j
            j += 1
        out.append((start, last_still, ended_by_loss))
        i = j + 1
    return out


@register
class StoppedVehicle(EventRule):
    label = "stopped_vehicle"

    def detect(self, tracks, signal, meta):
        v_stop = self.p("stop_speed", 0.1)
        v_go = self.p("go_speed", 0.3)
        go_sec = self.p("go_sec", 1.0)
        min_sec = self.p("min_stationary_sec", 10.0)
        moving = self.p("moving_speed", 0.5)
        need_move = self.p("require_moving_before", True)
        lone_min = self.p("lone_min_sec", 60.0)
        radius_bl = self.p("pass_radius_bl", 4.0)
        link_gap = self.p("link_gap_sec", 5.0)
        link_iou = self.p("link_iou", 0.6)

        moved_bl = self.p("moved_before_bl", 3.0)
        veh_all = tracks.of_kind("vehicle")
        veh = tracks.of_kind("vehicle", reliable=True)
        if veh.empty:
            return []
        eps = []
        for tid, g in veh.groupby("track_id", sort=True):
            t = g["t"].to_numpy()
            sp = g["speed"].to_numpy()
            sx, sy = g["sx"].to_numpy(), g["sy"].to_numpy()
            for s, e, lost in stationary_episodes(t, sp, v_stop, v_go, go_sec):
                before = (t < t[s]) & (t >= t[s] - 15.0) & (sp > moving)
                # Real motion before the stop: net displacement, not speed noise on a parked car.
                w = np.flatnonzero((t < t[s]) & (t >= t[s] - 15.0))
                disp = np.hypot(sx[s] - sx[w[0]], sy[s] - sy[w[0]]) / float(tracks.scale(sy[s])) if len(w) else 0.0
                moved = bool(disp >= moved_bl)
                heading = None
                if moved and before.any():
                    k = np.flatnonzero(before)[-5:]
                    vx, vy = g["vx"].to_numpy()[k].mean(), g["vy"].to_numpy()[k].mean()
                    heading = np.array([vx, vy]) / max(np.hypot(vx, vy), 1e-9)
                eps.append(
                    {
                        "tid": tid,
                        "start": t[s],
                        "end": t[e],
                        "lost": lost,
                        "x": float(np.median(g["foot_x"].to_numpy()[s : e + 1])),
                        "y": float(np.median(g["foot_y"].to_numpy()[s : e + 1])),
                        "box": g[["x1", "y1", "x2", "y2"]].to_numpy()[s : e + 1].mean(0),
                        "moved": moved,
                        "heading": heading,
                        "tids": {tid},
                    }
                )

        eps = self._link(eps, link_gap, link_iou)
        vt = veh_all["t"].to_numpy()
        vfx = veh_all["foot_x"].to_numpy()
        vfy = veh_all["foot_y"].to_numpy()
        vsp = veh_all["speed"].to_numpy()
        vvx = veh_all["vx"].to_numpy()
        vvy = veh_all["vy"].to_numpy()
        vtid = veh_all["track_id"].to_numpy()
        out = []
        for ep in eps:
            dur = ep["end"] - ep["start"]
            if dur < min_sec or (need_move and not ep["moved"]):
                continue
            if self.scene.has_carriageway() and not self.scene.in_carriageway(np.array([ep["x"]]), np.array([ep["y"]]))[0]:
                continue
            if self.scene.queue_zones and points_in_any(np.array([ep["x"]]), np.array([ep["y"]]), self.scene.queue_zones)[0]:
                mid = (ep["start"] + ep["end"]) / 2
                if signal.known() and signal.state_at(None, mid) != "green":
                    continue
            during = (vt >= ep["start"]) & (vt <= ep["end"]) & ~np.isin(vtid, list(ep["tids"]))
            r = radius_bl * float(tracks.scale(ep["y"]))
            near = during & (np.hypot(vfx - ep["x"], vfy - ep["y"]) < r) & (vsp > moving)
            direction = ep["heading"]
            if direction is None:
                d, st = self.scene.lane_direction(np.array([ep["x"]]), np.array([ep["y"]]))
                direction = d[0] if st[0] > 0.3 else None
            if direction is not None and near.any():
                u = np.stack([vvx[near], vvy[near]], 1) / np.maximum(vsp[near], 1e-9)[:, None]
                same = (u @ direction) > 0.5
                passers = len(set(vtid[near][same].tolist()))
            else:
                passers = len(set(vtid[near].tolist()))
            others = during & (np.hypot(vfx - ep["x"], vfy - ep["y"]) < 3 * r)
            jam = others.sum() >= 10 and float(np.median(vsp[others])) < 0.3
            if jam:
                conf = 0.2
            elif passers >= 1:
                conf = 0.9
            elif dur >= lone_min:
                conf = 0.7
            else:
                conf = 0.3
            out.append(Segment(ep["start"], ep["end"], self.label, conf, {"tracks": sorted(ep["tids"]), "passers": passers}))
        return out

    @staticmethod
    def _link(eps: list[dict], gap: float, iou_thr: float) -> list[dict]:
        """Join episodes of different track IDs at the same place (ID switch while stationary)."""
        eps = sorted(eps, key=lambda e: e["start"])
        merged: list[dict] = []
        for ep in eps:
            joined = False
            for m in merged:
                if m["lost"] and 0 <= ep["start"] - m["end"] <= gap and box_iou(m["box"], ep["box"])[0, 0] > iou_thr:
                    m["end"] = max(m["end"], ep["end"])
                    m["lost"] = ep["lost"]
                    m["tids"] |= ep["tids"]
                    joined = True
                    break
            if not joined:
                merged.append(ep)
        return merged
