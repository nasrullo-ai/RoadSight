"""congestion: many vehicles of one travel direction nearly stopped for a sustained period."""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration


@register
class Congestion(EventRule):
    label = "congestion"

    def detect(self, tracks, signal, meta):
        bin_sec = self.p("bin_sec", 1.0)
        v_thr = self.p("speed_thr", 0.2)
        min_veh = self.p("min_vehicles", 6)
        cap_frac = self.p("capacity_frac", 0.6)
        min_len = self.p("min_len_sec", 15.0)
        min_len_nosig = self.p("min_len_no_signal_sec", 60.0)
        green_overlap = self.p("green_overlap_sec", 5.0)
        moving = self.p("moving_speed", 0.5)

        veh = tracks.of_kind("vehicle")
        if veh.empty:
            return []
        if self.scene.has_carriageway():
            veh = veh[self.scene.in_carriageway(veh["foot_x"].to_numpy(), veh["foot_y"].to_numpy())]
        if veh.empty:
            return []

        # Reference travel direction per track (mean while moving, else lane flow).
        ang = {}
        for tid, g in veh.groupby("track_id", sort=True):
            mv = g[g["speed"] > moving]
            if len(mv):
                ang[tid] = float(np.degrees(np.arctan2(mv["vy"].mean(), mv["vx"].mean())))
            else:
                d, st = self.scene.lane_direction(g["foot_x"].to_numpy()[:1], g["foot_y"].to_numpy()[:1])
                if st[0] > 0.3:
                    ang[tid] = float(np.degrees(np.arctan2(d[0, 1], d[0, 0])))
        if not ang:
            groups = {tid: 0 for tid in veh["track_id"].unique()}
        else:
            hist, edges = np.histogram(list(ang.values()), bins=36, range=(-180, 180))
            axis = edges[np.argmax(hist)] + 5.0
            groups = {tid: int(((a - axis + 45.0) % 360.0) // 90.0) for tid, a in ang.items()}
        gid = veh["track_id"].map(groups).fillna(-1).astype(int).to_numpy()

        n_bins = int(np.ceil(max(meta.duration, veh["t"].max() + bin_sec) / bin_sec))
        b = np.clip((veh["t"].to_numpy() / bin_sec).astype(int), 0, n_bins - 1)
        tid = veh["track_id"].to_numpy()
        sp = veh["speed"].to_numpy()
        congested = np.zeros(n_bins, dtype=bool)
        for g in sorted(set(gid.tolist()) - {-1}):
            m = gid == g
            counts = np.zeros(n_bins)
            meds = np.full(n_bins, np.inf)
            order = np.lexsort((tid[m], b[m]))
            bb, tt, ss = b[m][order], tid[m][order], sp[m][order]
            for k in np.unique(bb):
                sel = bb == k
                per_track = {}
                for t_, s_ in zip(tt[sel], ss[sel]):
                    per_track.setdefault(t_, []).append(s_)
                counts[k] = len(per_track)
                meds[k] = float(np.median([np.median(v) for v in per_track.values()]))
            cap = np.percentile(counts[counts > 0], 95) if (counts > 0).any() else 0
            need = max(min_veh, cap_frac * cap)
            congested |= (counts >= need) & (meds < v_thr)

        times = np.arange(n_bins) * bin_sec
        out = []
        sig_known = signal.known()
        for s, e in runs_min_duration(congested, times, min_len, max_gap_sec=3 * bin_sec):
            start, end = times[s], times[e] + bin_sec
            if sig_known:
                green = sum(
                    max(0.0, min(end, b_) - max(start, a_)) for segs in signal.segments.values() for a_, b_, st in segs if st == "green"
                )
                if green < green_overlap:
                    continue
            elif end - start < min_len_nosig:
                continue
            out.append(Segment(start, end, self.label, 0.8))
        return out
