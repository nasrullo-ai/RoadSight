"""accident: contact between road users with an abrupt speed drop or heading jump, confirmed by
post-event stationarity or people appearing next to the vehicles."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.perception.interactions import window_mean
from roadsight.scene.geometry import angle_diff_deg, box_iou, points_in_any


def abrupt_moments(t, sp, heading, v_min, ratio, dv_min, heading_jump, gap=0.2, win=1.0) -> list[tuple[int, str]]:
    """First index of each run where the speed collapses or the heading jumps."""
    before = window_mean(t, sp, t - win, t - gap)
    after = window_mean(t, sp, t + gap, t + win)
    drop = (before >= v_min) & (after <= ratio * before) & ((before - after) >= dv_min)
    hx, hy = np.cos(np.radians(heading)) * sp, np.sin(np.radians(heading)) * sp
    hb = np.degrees(np.arctan2(window_mean(t, hy, t - win, t - gap), window_mean(t, hx, t - win, t - gap)))
    ha = np.degrees(np.arctan2(window_mean(t, hy, t + gap, t + win), window_mean(t, hx, t + gap, t + win)))
    turn = (before >= v_min) & (after >= 0.3) & (np.abs(angle_diff_deg(ha, hb)) > heading_jump)
    flag = np.nan_to_num(drop, nan=0).astype(bool) | np.nan_to_num(turn, nan=0).astype(bool)
    out = []
    prev = False
    for i, f in enumerate(flag):
        if f and not prev:
            out.append((i, "drop" if drop[i] else "turn"))
        prev = bool(f)
    return out


@register
class Accident(EventRule):
    label = "accident"

    def detect(self, tracks, signal, meta):
        v_min = self.p("v_min", 0.8)
        ratio = self.p("drop_ratio", 0.35)
        dv_min = self.p("dv_min", 0.8)
        heading_jump = self.p("heading_jump_deg", 40.0)
        iou_thr = self.p("contact_iou", 0.05)
        pre, post = self.p("contact_pre_sec", 1.0), self.p("contact_post_sec", 0.5)
        v_still = self.p("still_speed", 0.15)
        still_sec = self.p("still_sec", 3.0)
        confirm_win = self.p("confirm_window_sec", 8.0)
        person_win = self.p("person_window_sec", 20.0)
        person_bl = self.p("person_radius_bl", 3.0)
        cap = self.p("max_len_sec", 60.0)

        users = tracks.of_kind("road_user", reliable=True)
        if users.empty:
            return []
        per_track = {int(k): g for k, g in users.groupby("track_id", sort=True)}
        frame_rows = {int(f): g for f, g in users.groupby("frame", sort=True)}
        frame_t = np.array(sorted({float(v) for v in users["t"].unique()}))
        t_to_frame = dict(zip(users["t"].to_numpy(), users["frame"].to_numpy()))

        cands = []
        for tid, g in per_track.items():
            if g["cls"].iloc[0] == 0 and not g["rider"].iloc[0]:
                continue  # pedestrians are partners, not the decelerating body
            t = g["t"].to_numpy()
            for i, kind in abrupt_moments(t, g["speed"].to_numpy(), g["heading"].to_numpy(), v_min, ratio, dv_min, heading_jump):
                t0 = t[i]
                partner, tc = self._partner(tid, g, t0 - pre, t0 + post, frame_t, t_to_frame, frame_rows, iou_thr)
                fixed = False
                if partner is None and self.scene.fixed_objects:
                    fixed = bool(
                        points_in_any(g["foot_x"].to_numpy()[i : i + 1], g["foot_y"].to_numpy()[i : i + 1], self.scene.fixed_objects)[0]
                    )
                cands.append(
                    {
                        "tid": tid,
                        "partner": partner,
                        "tc": tc if tc is not None else t0,
                        "kind": kind,
                        "fixed": fixed,
                        "x": float(g["foot_x"].to_numpy()[i]),
                        "y": float(g["foot_y"].to_numpy()[i]),
                    }
                )

        persons = tracks.of_kind("person")
        p_first = persons.groupby("track_id", sort=True).first() if not persons.empty else None
        out = []
        for c in sorted(cands, key=lambda c: c["tc"]):
            involved = [c["tid"]] + ([c["partner"]] if c["partner"] is not None else [])
            tc = c["tc"]
            stationary = any(self._still_after(per_track[k], tc, v_still, still_sec, confirm_win) for k in involved)
            people = False
            if p_first is not None:
                r = person_bl * float(tracks.scale(c["y"]))
                m = (p_first["t"] >= tc) & (p_first["t"] <= tc + person_win)
                m &= np.hypot(p_first["foot_x"] - c["x"], p_first["foot_y"] - c["y"]) < r
                people = bool(m.any())
            if c["partner"] is not None or c["fixed"]:
                conf = 0.9 if (stationary and people) else 0.75 if stationary else 0.65 if people else 0.3
            else:
                conf = 0.55 if (stationary and people) else 0.35 if stationary else 0.1
            if c["kind"] == "turn" and not stationary:
                conf = min(conf, 0.3)
            end = self._end_time(per_track, involved, tc, v_still, cap)
            out.append(Segment(float(tc), float(max(end, tc + 1.0)), self.label, conf, {"tracks": involved, "kind": c["kind"]}))
        return out

    @staticmethod
    def _partner(tid, g, t_lo, t_hi, frame_t, t_to_frame, frame_rows, iou_thr):
        lo, hi = np.searchsorted(frame_t, t_lo), np.searchsorted(frame_t, t_hi, side="right")
        own = g.set_index("t")
        for tt in frame_t[lo:hi]:
            if tt not in own.index:
                continue
            rows = frame_rows.get(int(t_to_frame[tt]))
            others = rows[rows["track_id"] != tid]
            if others.empty:
                continue
            box = own.loc[tt, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
            iou = box_iou(box, others[["x1", "y1", "x2", "y2"]].to_numpy())[0]
            k = int(np.argmax(iou))
            if iou[k] > iou_thr:
                return int(others["track_id"].iloc[k]), float(tt)
        return None, None

    @staticmethod
    def _still_after(g, tc, v_still, still_sec, window) -> bool:
        t = g["t"].to_numpy()
        sp = g["speed"].to_numpy()
        m = (t >= tc) & (t <= tc + window)
        if m.sum() < 2:
            return False
        tt, ss = t[m], sp[m]
        run_start = None
        for a, s in zip(tt, ss):
            if s < v_still:
                run_start = a if run_start is None else run_start
                if a - run_start >= still_sec:
                    return True
            else:
                run_start = None
        # Track ends while slow and still visible for most of the window -> treat as stopped.
        return bool(run_start is not None and tt[-1] - run_start >= still_sec * 0.5 and t[-1] <= tc + window)

    @staticmethod
    def _end_time(per_track, involved, tc, v_still, cap) -> float:
        ends = []
        for k in involved:
            g = per_track[k]
            t = g["t"].to_numpy()
            sp = g["speed"].to_numpy()
            m = t >= tc
            if not m.any():
                ends.append(tc)
                continue
            tt, ss = t[m], sp[m]
            end = tt[-1]
            still_from = None
            for a, s in zip(tt, ss):
                if s < v_still:
                    still_from = a if still_from is None else still_from
                    if a - still_from >= 1.0:
                        end = still_from
                        break
                else:
                    still_from = None
            ends.append(end)
        return min(max(ends), tc + cap)
