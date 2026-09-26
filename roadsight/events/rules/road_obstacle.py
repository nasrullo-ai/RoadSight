"""road_obstacle: an animal or dropped object on the carriageway, or (optional) a static
foreground blob on the carriageway that no tracked road user explains."""

from __future__ import annotations

import cv2
import numpy as np

from roadsight.events.base import EventRule, Segment, register, runs_min_duration

ANIMALS = (16, 17, 18, 19)  # dog, horse, sheep, cow
OBJECTS = (28,)  # suitcase; backpacks and handbags are almost always carried


@register
class RoadObstacle(EventRule):
    label = "road_obstacle"

    def detect(self, tracks, signal, meta):
        out = self._from_classes(tracks)
        if self.p("blob_enabled", False) and getattr(tracks, "bg_samples", None):
            out += self._from_blobs(tracks)
        return out

    def _from_classes(self, tracks) -> list[Segment]:
        if not self.scene.has_carriageway():
            return []
        animal_sec = self.p("animal_min_sec", 3.0)
        object_sec = self.p("object_min_sec", 5.0)
        df = tracks.df[tracks.df["cls"].isin(ANIMALS + OBJECTS)]
        if df.empty:
            return []
        # Objects or animals next to a person are carried or walked, not obstacles.
        people = tracks.df[tracks.df["cls"] == 0]
        p_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in people.groupby("frame", sort=True)}
        pad = self.p("person_pad_bl", 1.0)
        near = []
        for r in df.itertuples(index=False):
            b = p_by_frame.get(int(r.frame))
            s = pad * float(tracks.scale(r.foot_y))
            near.append(
                b is not None
                and bool(
                    ((r.foot_x >= b[:, 0] - s) & (r.foot_x <= b[:, 2] + s) & (r.foot_y >= b[:, 1] - s) & (r.foot_y <= b[:, 3] + s)).any()
                )
            )
        df = df[~np.array(near, dtype=bool)]
        out = []
        for tid, g in df.groupby("track_id", sort=True):
            t = g["t"].to_numpy()
            on = self.scene.in_carriageway(g["foot_x"].to_numpy(), g["foot_y"].to_numpy())
            if g["cls"].iloc[0] in ANIMALS:
                runs_ = runs_min_duration(on, t, animal_sec, max_gap_sec=1.0)
            else:
                runs_ = runs_min_duration(on & (g["speed"].to_numpy() < 0.3), t, object_sec, max_gap_sec=1.0)
            for s, e in runs_:
                out.append(Segment(float(t[s]), float(t[e]), self.label, 0.65, {"tracks": [int(tid)], "cls": int(g["cls"].iloc[0])}))
        return out

    def _from_blobs(self, tracks) -> list[Segment]:
        """Background subtraction on 1 Hz grey thumbnails; the background is the median of 10-60 s ago."""
        samples = tracks.bg_samples  # list of (t, gray uint8 HxW)
        if len(samples) < 20:
            return []
        diff_thr = self.p("blob_diff", 30)
        persist = self.p("blob_min_sec", 5.0)
        min_area = self.p("blob_min_area_frac", 0.0015)
        ts = np.array([t for t, _ in samples])
        imgs = np.stack([im for _, im in samples])
        h, w = imgs.shape[1:]
        sx, sy = w / tracks.width, h / tracks.height
        yy, xx = np.mgrid[0:h, 0:w]
        road = self.scene.in_carriageway((xx + 0.5) / sx, (yy + 0.5) / sy)
        boxes_by_t = {}
        for _, g in tracks.df.groupby("frame", sort=True):
            boxes_by_t[float(g["t"].iloc[0])] = g[["x1", "y1", "x2", "y2"]].to_numpy()
        box_times = np.array(sorted(boxes_by_t))
        kernel = np.ones((3, 3), np.uint8)
        fg_seq = []
        for k, t in enumerate(ts):
            hist = (ts >= t - 60) & (ts <= t - 10)
            if hist.sum() < 8:
                fg_seq.append(np.zeros((h, w), bool))
                continue
            bg = np.median(imgs[hist], axis=0)
            fg = (np.abs(imgs[k].astype(np.int16) - bg.astype(np.int16)) > diff_thr).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = fg.astype(bool) & road
            if len(box_times):
                j = int(np.argmin(np.abs(box_times - t)))
                if abs(box_times[j] - t) < 1.0:
                    for x1, y1, x2, y2 in boxes_by_t[box_times[j]]:
                        pad = 0.2 * (x2 - x1)
                        fg[
                            max(0, int((y1 - pad) * sy)) : int((y2 + pad) * sy) + 1, max(0, int((x1 - pad) * sx)) : int((x2 + pad) * sx) + 1
                        ] = False
            fg_seq.append(fg)
        fg_seq = np.stack(fg_seq)
        n_hold = max(2, int(round(persist / max(np.median(np.diff(ts)), 1e-3))))
        static = np.zeros(len(ts), dtype=bool)
        for k in range(n_hold - 1, len(ts)):
            common = np.logical_and.reduce(fg_seq[k - n_hold + 1 : k + 1])
            static[k] = common.sum() >= min_area * h * w
        out = []
        for s, e in runs_min_duration(static, ts, 0.0, max_gap_sec=2.0):
            start = ts[max(0, s - n_hold + 1)]
            out.append(Segment(float(start), float(ts[e]), self.label, self.p("blob_confidence", 0.5), {"source": "blob"}))
        return out
