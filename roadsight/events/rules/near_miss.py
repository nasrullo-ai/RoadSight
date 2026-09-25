"""near_miss: a pair on a collision course (TTC < 1 s) with hard braking or a swerve, and no contact.

Evasive-action thresholds adapt to each video's kinematic noise (MAD of acceleration, upper
percentile of heading change), so noisy far-field tracks in dense traffic do not fire.
"""

from __future__ import annotations

import numpy as np

from roadsight.events.base import EventRule, Segment, register
from roadsight.perception.interactions import candidate_pairs, overlap, pair_ttc, window_mean
from roadsight.scene.geometry import angle_diff_deg, footprint


@register
class NearMiss(EventRule):
    label = "near_miss"

    def detect(self, tracks, signal, meta):
        ttc_thr = self.p("ttc_thr", 1.0)
        a_brake = self.p("brake_decel", 1.5)
        swerve_deg = self.p("swerve_deg", 25.0)
        k_sigma = self.p("brake_k_sigma", 4.0)
        swerve_pct = self.p("swerve_percentile", 99.5)
        max_dist_bl = self.p("max_dist_bl", 6.0)
        min_speed = self.p("min_speed", 0.5)
        min_closing = self.p("min_closing_bl", 1.0)
        min_hits = int(self.p("min_hits", 2))
        min_size = self.p("min_size_frac", 0.035) * tracks.height
        clear_ttc = self.p("clear_ttc", 3.0)
        max_len = self.p("max_len_sec", 6.0)
        horizon = self.p("horizon_sec", 3.0)
        max_speed = self.p("max_plausible_speed_bl", 8.0)
        min_real = self.p("min_real_frac", 0.7)
        brake_drop = self.p("brake_drop_ratio", 0.6)

        users = tracks.of_kind("road_user", reliable=True)
        users = users[((users["cls"] != 0) | users["rider"] | (users["speed"] > 0.2)) & (users["size"] >= min_size) & ~users["edge"]]
        if users["track_id"].nunique() < 2:
            return []
        span = users.groupby("track_id")["t"].agg(lambda s: s.max() - s.min())
        users = users[users["track_id"].isin(span.index[span >= self.p("min_track_sec", 1.5)])]
        if users["track_id"].nunique() < 2:
            return []
        per_track = {int(k): g for k, g in users.groupby("track_id", sort=True)}
        swerve = {k: self._swerve(g, min_speed) for k, g in per_track.items()}

        # Per-video noise floor for evasive manoeuvres.
        mv = users[users["speed"] > min_speed]
        acc = mv["accel"].to_numpy()
        if len(acc) > 50:
            sigma = 1.4826 * float(np.median(np.abs(acc - np.median(acc))))
            a_brake = max(a_brake, k_sigma * sigma)
        sw_all = np.concatenate([v[v > 0] for v in swerve.values()]) if swerve else np.zeros(0)
        if len(sw_all) > 50:
            swerve_deg = max(swerve_deg, float(np.percentile(sw_all, swerve_pct)))

        hits: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for _, g in users.groupby("frame", sort=True):
            if len(g) < 2:
                continue
            sp = g["speed"].to_numpy()
            if (sp > min_speed).sum() == 0:
                continue
            x = g["sx"].to_numpy()
            y = g["sy"].to_numpy()
            scale = tracks.scale(y)
            pairs = candidate_pairs(x, y, max_dist_bl * scale)
            if len(pairs) == 0:
                continue
            pairs = pairs[(sp[pairs[:, 0]] > min_speed) | (sp[pairs[:, 1]] > min_speed)]
            vx, vy = g["vx"].to_numpy(), g["vy"].to_numpy()
            i, j = pairs[:, 0], pairs[:, 1]
            dx, dy = x[j] - x[i], y[j] - y[i]
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            rvx = vx[j] * scale[j] - vx[i] * scale[i]
            rvy = vy[j] * scale[j] - vy[i] * scale[i]
            closing = -(rvx * dx + rvy * dy) / dist / ((scale[i] + scale[j]) / 2)  # BL/s
            pairs = pairs[closing >= min_closing]
            if len(pairs) == 0:
                continue
            vel = np.stack([vx * scale, vy * scale], 1)
            ttc, now = pair_ttc(g[["x1", "y1", "x2", "y2"]].to_numpy(), vel, pairs, horizon)
            ids = g["track_id"].to_numpy()
            t = float(g["t"].iloc[0])
            for (a, b), v, o in zip(pairs, ttc, now):
                if o or v >= ttc_thr:
                    continue
                key = (int(min(ids[a], ids[b])), int(max(ids[a], ids[b])))
                hits.setdefault(key, []).append((t, float(v)))

        dt = float(np.median(np.diff(tracks.frame_times))) if len(tracks.frame_times) > 1 else 0.1
        out = []
        for (a, b), lst in sorted(hits.items()):
            ga, gb = per_track[a], per_track[b]
            times = np.array([x[0] for x in lst])
            ttcs = np.array([x[1] for x in lst])
            cuts = np.flatnonzero(np.diff(times) > 3.0) + 1
            for idx in np.split(np.arange(len(times)), cuts):
                if len(idx) < min_hits or not self._consecutive(times[idx], dt, min_hits):
                    continue
                t0 = times[idx[0]]
                if not (self._plausible(ga, t0, max_speed, min_real) and self._plausible(gb, t0, max_speed, min_real)):
                    continue  # implausible speeds or mostly interpolated: tracking artefact
                if self._contact(ga, gb, t0, t0 + 3.0):
                    continue  # contact -> accident, not near miss
                ev = [
                    x
                    for x in (
                        self._evasion(ga, swerve[a], t0, a_brake, swerve_deg, brake_drop),
                        self._evasion(gb, swerve[b], t0, a_brake, swerve_deg, brake_drop),
                    )
                    if x is not None
                ]
                if not ev:
                    continue
                if self._queue_join(ga, gb, t0):
                    continue  # braking behind a waiting queue where vehicles always stop: normal traffic
                start = min(x[0] for x in ev)
                severity = max(x[1] for x in ev)
                end = self._clear_time(ga, gb, times[idx[-1]], clear_ttc, horizon, tracks)
                end = min(end, start + max_len)
                tmin = float(ttcs[idx].min())
                conf = float(np.clip(0.45 + 0.3 * (1 - tmin / ttc_thr) + 0.15 * severity + 0.1 * (len(ev) - 1), 0, 0.95))
                out.append(Segment(float(start), float(max(end, start + 0.5)), self.label, conf, {"tracks": [a, b], "ttc": tmin}))
        return out

    @staticmethod
    def _consecutive(times: np.ndarray, dt: float, n: int) -> bool:
        run = 1
        for d in np.diff(times):
            run = run + 1 if d <= 1.5 * dt else 1
            if run >= n:
                return True
        return n <= 1

    @staticmethod
    def _swerve(g, min_speed) -> np.ndarray:
        t = g["t"].to_numpy()
        h = g["heading"].to_numpy()
        sp = g["speed"].to_numpy()
        j = np.clip(np.searchsorted(t, t - 1.0), 0, len(t) - 1)
        d = np.abs(angle_diff_deg(h, h[j]))
        return np.where((sp > min_speed) & (sp[j] > min_speed) & (t - t[j] >= 0.5), d, 0.0)

    def _queue_join(self, ga, gb, t0) -> bool:
        """A vehicle in a learned queue zone and the slower one (nearly) standing: joining a queue."""
        auto = self.scene.auto
        if self.scene.queue_zones:
            from roadsight.scene.geometry import points_in_any

            def in_queue(x, y):
                return bool(points_in_any(np.array([x]), np.array([y]), self.scene.queue_zones)[0])

        elif auto is not None and auto.queue is not None:

            def in_queue(x, y):
                return bool(auto.lookup(auto.queue, np.array([x]), np.array([y]))[0])

        else:
            return False
        rows = []
        for g in (ga, gb):
            w = g[(g["t"] >= t0 - 0.5) & (g["t"] <= t0 + 0.5)]
            if w.empty:
                return False
            rows.append((float(w["sx"].median()), float(w["sy"].median()), float(w["speed"].median())))
        any_in = any(in_queue(x, y) for x, y, _ in rows)
        return any_in and min(sp for _, _, sp in rows) < self.p("queue_leader_speed", 0.5)

    @staticmethod
    def _plausible(g, t0, max_speed, min_real) -> bool:
        w = g[(g["t"] >= t0 - 2.0) & (g["t"] <= t0 + 2.0)]
        return len(w) > 0 and float(w["speed"].max()) <= max_speed and float((~w["interp"]).mean()) >= min_real

    @staticmethod
    def _evasion(g, swerve, t0, a_brake, swerve_deg, brake_drop=0.6) -> tuple[float, float] | None:
        """(onset time, severity 0..1) of hard braking or swerving around ``t0``.

        Braking counts only when the speed over the second before was >= 1 BL/s and the speed over
        the second after falls below ``brake_drop`` of it: a real manoeuvre, not a noise spike.
        """
        t = g["t"].to_numpy()
        acc = g["accel"].to_numpy()
        sp = g["speed"].to_numpy()
        before = window_mean(t, sp, t - 1.0, t - 0.2)
        after = window_mean(t, sp, t + 0.2, t + 1.0)
        real_brake = (acc < -a_brake) & np.nan_to_num((before >= 1.0) & (after <= brake_drop * before), nan=0).astype(bool)
        m = (t >= t0 - 0.5) & (t <= t0 + 1.5)
        ev = m & (real_brake | (swerve > swerve_deg))
        if not ev.any():
            return None
        sev = max(float(np.max(-acc[m])) / a_brake - 1.0, float(np.max(swerve[m])) / swerve_deg - 1.0)
        return float(t[ev][0]), float(np.clip(sev, 0.0, 1.0))

    @staticmethod
    def _contact(ga, gb, t_lo, t_hi) -> bool:
        a = ga[(ga["t"] >= t_lo) & (ga["t"] <= t_hi)].set_index("frame")
        b = gb[(gb["t"] >= t_lo) & (gb["t"] <= t_hi)].set_index("frame")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return False
        fa = footprint(a.loc[common, ["x1", "y1", "x2", "y2"]].to_numpy())
        fb = footprint(b.loc[common, ["x1", "y1", "x2", "y2"]].to_numpy())
        return bool(overlap(fa, fb).any())

    @staticmethod
    def _clear_time(ga, gb, t_last, clear_ttc, horizon, tracks) -> float:
        a = ga[ga["t"] >= t_last].set_index("frame")
        b = gb[gb["t"] >= t_last].set_index("frame")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return float(t_last)
        a, b = a.loc[common], b.loc[common]
        d = np.hypot(a["sx"].to_numpy() - b["sx"].to_numpy(), a["sy"].to_numpy() - b["sy"].to_numpy())
        boxes = np.vstack([a[["x1", "y1", "x2", "y2"]].to_numpy(), b[["x1", "y1", "x2", "y2"]].to_numpy()])
        n = len(a)
        sa, sb = tracks.scale(a["sy"].to_numpy()), tracks.scale(b["sy"].to_numpy())
        vel = np.vstack([np.stack([a["vx"] * sa, a["vy"] * sa], 1), np.stack([b["vx"] * sb, b["vy"] * sb], 1)])
        pairs = np.stack([np.arange(n), np.arange(n) + n], 1)
        ttc, _ = pair_ttc(boxes, vel, pairs, horizon)
        t = a["t"].to_numpy()
        inc = np.concatenate([[False], np.diff(d) > 0])
        ok = inc & (ttc > clear_ttc)
        return float(t[ok][0]) if ok.any() else float(t[-1])
