"""TrackTable: one row per (track, processed frame) with boxes and kinematics.

Columns: track_id, frame, t, cls, conf, x1, y1, x2, y2, foot_x, foot_y, size, vx, vy, speed,
heading, accel, rider, interp. Speeds are in BL/s (see kinematics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from roadsight.perception.kinematics import VEHICLE_CLASSES, RowScale, smooth_track
from roadsight.scene.geometry import box_iou

PERSON = 0
TWO_WHEELERS = (1, 3)
RAW_COLUMNS = ["frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]


class TrackTable:
    def __init__(self, df: pd.DataFrame, scale: RowScale, frame_times: np.ndarray, width: int, height: int) -> None:
        self.df = df
        self.scale = scale
        self.frame_times = frame_times  # times of all processed frames, sorted
        self.width = width
        self.height = height
        self._by_track = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_records(
        cls, records: np.ndarray, frame_times: np.ndarray, frame_ids: np.ndarray, width: int, height: int, cfg: dict
    ) -> TrackTable:
        """``records`` rows: frame, t, track_id, cls, conf, x1, y1, x2, y2."""
        if len(records) == 0:
            return cls(empty_frame(), RowScale.default(height), np.asarray(frame_times, float), width, height)
        df = pd.DataFrame(np.asarray(records, dtype=np.float64), columns=RAW_COLUMNS)
        df["track_id"] = df["track_id"].astype(np.int64)
        df["frame"] = df["frame"].astype(np.int64)
        df = df.sort_values(["track_id", "frame"], kind="mergesort").drop_duplicates(["track_id", "frame"])
        df["cls"] = majority_class(df)
        df["interp"] = False
        df["foot_x"] = (df["x1"] + df["x2"]) / 2.0
        df["foot_y"] = df["y2"]
        df["size"] = np.sqrt(np.clip(df["x2"] - df["x1"], 1, None) * np.clip(df["y2"] - df["y1"], 1, None))
        # Boxes cut by any border have a wrong anchor or ground point.
        df["edge"] = border_rows(df, width, height)

        # Split ID switches on the raw detections, then interpolate short gaps inside each piece.
        scale = fit_scale(df, width, height)
        max_gap = float(cfg.get("max_interp_gap_sec", 1.0))
        df = split_tracks(df, scale, max_gap, float(cfg.get("max_jump_speed_bl", 20.0)), float(cfg.get("max_size_ratio", 1.8)))
        df["cls"] = majority_class(df)
        df = interpolate_gaps(df[RAW_COLUMNS], np.asarray(frame_ids), np.asarray(frame_times), max_gap)
        df["foot_x"] = (df["x1"] + df["x2"]) / 2.0
        df["foot_y"] = df["y2"]
        df["size"] = np.sqrt(np.clip(df["x2"] - df["x1"], 1, None) * np.clip(df["y2"] - df["y1"], 1, None))
        df["edge"] = border_rows(df, width, height)

        # Drop short tracks.
        span = df.groupby("track_id")["t"].agg(["min", "max"])
        keep = span.index[(span["max"] - span["min"]) >= float(cfg.get("min_track_sec", 1.0))]
        df = df[df["track_id"].isin(keep)].reset_index(drop=True)

        # Track-level reliability: rules ignore faint or tiny tracks, which carry most of the noise.
        real = df[~df["interp"]]
        tconf = real.groupby("track_id")["conf"].mean()
        tsize = df.groupby("track_id")["size"].median()
        df["track_conf"] = df["track_id"].map(tconf).fillna(0.0).to_numpy()
        df["track_size"] = df["track_id"].map(tsize).to_numpy()
        df["reliable"] = (df["track_conf"] >= float(cfg.get("reliable_min_conf", 0.4))) & (
            df["track_size"] >= float(cfg.get("reliable_min_size_frac", 0.025)) * height
        )

        df["rider"] = mark_riders(df)
        df = add_kinematics(df, scale, float(cfg.get("savgol_window_sec", 1.0)), int(cfg.get("savgol_order", 2)))
        df = df.sort_values(["track_id", "t"], kind="mergesort").reset_index(drop=True)
        return cls(df, scale, np.asarray(frame_times, float), width, height)

    # ------------------------------------------------------------------ access
    def tracks(self):
        """Iterate ``(track_id, DataFrame)`` in track-id order."""
        if self._by_track is None:
            self._by_track = [(int(k), g) for k, g in self.df.groupby("track_id", sort=True)]
        return self._by_track

    def of_kind(self, kind: str, reliable: bool = False) -> pd.DataFrame:
        df = self.df
        if reliable and "reliable" in df:
            df = df[df["reliable"]]
        if kind == "vehicle":
            return df[df["cls"].isin(VEHICLE_CLASSES)]
        if kind == "person":
            return df[(df["cls"] == PERSON) & (~df["rider"])]
        if kind == "road_user":
            return df[df["cls"].isin((0, 1, 2, 3, 5, 7))]
        raise ValueError(kind)

    def by_frame(self) -> dict[int, pd.DataFrame]:
        return {int(k): g for k, g in self.df.groupby("frame", sort=True)}

    def __len__(self) -> int:
        return len(self.df)


def empty_frame() -> pd.DataFrame:
    cols = RAW_COLUMNS + [
        "foot_x",
        "foot_y",
        "size",
        "rider",
        "interp",
        "edge",
        "reliable",
        "track_conf",
        "track_size",
        "sx",
        "sy",
        "vx",
        "vy",
        "speed",
        "heading",
        "accel",
    ]
    df = pd.DataFrame({c: pd.Series(dtype=np.float64) for c in cols})
    df["track_id"] = df["track_id"].astype(np.int64)
    df["frame"] = df["frame"].astype(np.int64)
    df["cls"] = df["cls"].astype(np.int64)
    df["rider"] = df["rider"].astype(bool)
    df["interp"] = df["interp"].astype(bool)
    df["edge"] = df["edge"].astype(bool)
    df["reliable"] = df["reliable"].astype(bool)
    return df


def border_rows(df: pd.DataFrame, width: int, height: int, frac: float = 0.005) -> pd.Series:
    """Boxes touching (within 0.5% of) the frame border are cut off: their anchor or ground point is wrong."""
    mx, my = max(2.0, frac * width), max(2.0, frac * height)
    return (df["y1"] <= my) | (df["y2"] >= height - my) | (df["x1"] <= mx) | (df["x2"] >= width - mx)


def majority_class(df: pd.DataFrame) -> np.ndarray:
    """Confidence-weighted majority class per track, broadcast to rows."""
    w = df["conf"].where(df["conf"] > 0, 0.01)
    votes = w.groupby([df["track_id"], df["cls"]], sort=True).sum().rename("w").reset_index()
    votes = votes.sort_values(["track_id", "w", "cls"], ascending=[True, False, True], kind="mergesort")
    major = votes.drop_duplicates("track_id").set_index("track_id")["cls"]
    return df["track_id"].map(major).astype(np.int64).to_numpy()


def split_tracks(df: pd.DataFrame, scale: RowScale, max_gap_sec: float, max_speed_bl: float, max_size_ratio: float) -> pd.DataFrame:
    """Give a new id to each piece of a track after a long gap or an implausible jump (ID switch)."""
    df = df.sort_values(["track_id", "t"], kind="mergesort").reset_index(drop=True)
    tid = df["track_id"].to_numpy()
    t = df["t"].to_numpy()
    fx = df["foot_x"].to_numpy()
    fy = df["foot_y"].to_numpy()
    size = df["size"].to_numpy()
    edge = df["edge"].to_numpy()
    same = tid[1:] == tid[:-1]
    dt = np.maximum(np.diff(t), 1e-3)
    disp = np.hypot(np.diff(fx), np.diff(fy)) / scale(fy[1:])
    ratio = np.maximum(size[1:] / size[:-1], size[:-1] / size[1:])
    interior = ~edge[1:] & ~edge[:-1]
    jump = interior & (((disp / dt) > max_speed_bl) & (disp > 1.0) | (ratio > max_size_ratio))
    brk = same & ((dt > max_gap_sec + 1e-6) | jump)
    new_piece = np.ones(len(df), dtype=bool)
    new_piece[1:] = brk | ~same
    df["track_id"] = np.cumsum(new_piece).astype(np.int64)
    return df


def interpolate_gaps(df: pd.DataFrame, frame_ids: np.ndarray, frame_times: np.ndarray, max_gap_sec: float) -> pd.DataFrame:
    """Linearly fill processed frames missing inside a track when the gap is short."""
    df = df.copy()
    df["interp"] = False
    if max_gap_sec <= 0 or len(frame_ids) == 0:
        return df
    pos_of = {int(f): i for i, f in enumerate(frame_ids)}
    extra = []
    for tid, g in df.groupby("track_id", sort=True):
        frames = g["frame"].to_numpy()
        idx = np.array([pos_of.get(int(f), -1) for f in frames])
        if (idx < 0).any():
            continue
        gaps = np.flatnonzero(np.diff(idx) > 1)
        if len(gaps) == 0:
            continue
        vals = g[["t", "x1", "y1", "x2", "y2"]].to_numpy()
        for k in gaps:
            t0, t1 = vals[k, 0], vals[k + 1, 0]
            if t1 - t0 > max_gap_sec:
                continue
            for j in range(idx[k] + 1, idx[k + 1]):
                tt = frame_times[j]
                a = (tt - t0) / max(t1 - t0, 1e-9)
                box = vals[k, 1:] * (1 - a) + vals[k + 1, 1:] * a
                extra.append([frame_ids[j], tt, tid, g["cls"].iloc[0], 0.0, *box])
    if extra:
        add = pd.DataFrame(extra, columns=RAW_COLUMNS)
        add["track_id"] = add["track_id"].astype(np.int64)
        add["frame"] = add["frame"].astype(np.int64)
        add["cls"] = add["cls"].astype(np.int64)
        add["interp"] = True
        df = pd.concat([df, add], ignore_index=True)
    return df.sort_values(["track_id", "frame"], kind="mergesort").reset_index(drop=True)


def mark_riders(df: pd.DataFrame, min_frac: float = 0.3) -> np.ndarray:
    """Person tracks that mostly overlap a bicycle/motorcycle are riders, not pedestrians."""
    is_person = (df["cls"] == PERSON).to_numpy()
    is_2w = df["cls"].isin(TWO_WHEELERS).to_numpy()
    rider_row = np.zeros(len(df), dtype=bool)
    if not is_person.any() or not is_2w.any():
        return rider_row
    frames = df["frame"].to_numpy()
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy()
    rows_p = np.flatnonzero(is_person)
    rows_w = np.flatnonzero(is_2w)
    wf = frames[rows_w]
    order = np.argsort(wf, kind="mergesort")
    rows_w, wf = rows_w[order], wf[order]
    for f in np.unique(frames[rows_p]):
        lo, hi = np.searchsorted(wf, f), np.searchsorted(wf, f, side="right")
        if lo == hi:
            continue
        pr = rows_p[frames[rows_p] == f]
        pb = boxes[pr]
        wb = boxes[rows_w[lo:hi]]
        iou = box_iou(pb, wb)
        # A rider's lower body overlaps the two-wheeler box.
        cx = (pb[:, 0] + pb[:, 2]) / 2
        cy = pb[:, 1] + 0.75 * (pb[:, 3] - pb[:, 1])
        inside = (
            (cx[:, None] >= wb[None, :, 0])
            & (cx[:, None] <= wb[None, :, 2])
            & (cy[:, None] >= wb[None, :, 1] - 0.2 * (wb[None, :, 3] - wb[None, :, 1]))
            & (cy[:, None] <= wb[None, :, 3])
        )
        rider_row[pr] = (iou > 0.1).any(1) | inside.any(1)
    frac = pd.Series(rider_row, index=df.index).groupby(df["track_id"]).transform("mean").to_numpy()
    return is_person & (frac >= min_frac)


def fit_scale(df: pd.DataFrame, width: int, height: int) -> RowScale:
    v = df[df["cls"].isin(VEHICLE_CLASSES) & (~df["interp"])]
    if len(v) == 0:
        return RowScale.default(height)
    m = (v["x1"] > 2) & (v["y1"] > 2) & (v["x2"] < width - 2) & (v["y2"] < height - 2)
    cars = v[m & (v["cls"] == 2)]
    use = cars if len(cars) >= 30 else v[m]
    if len(use) == 0:
        use = v
    return RowScale.fit(use["foot_y"].to_numpy(), use["size"].to_numpy(), height)


def add_kinematics(df: pd.DataFrame, scale: RowScale, window_sec: float, order: int) -> pd.DataFrame:
    df = df.sort_values(["track_id", "t"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    names = ("sx", "sy", "vx", "vy", "speed", "heading", "accel")
    cols = {k: np.zeros(n) for k in names}
    tid = df["track_id"].to_numpy()
    t = df["t"].to_numpy()
    fx = df["foot_x"].to_numpy()
    y1 = df["y1"].to_numpy()
    h = (df["y2"] - df["y1"]).to_numpy()
    # Ground point = box top + rolling median height: robust to the bottom being occluded.
    h_med = (
        df.assign(h=h)
        .groupby("track_id", sort=False)["h"]
        .transform(lambda s: s.rolling(9, center=True, min_periods=1).median())
        .to_numpy()
    )
    fy = y1 + h_med
    edge = df["edge"].to_numpy() if "edge" in df else np.zeros(n, dtype=bool)
    bounds = np.flatnonzero(np.diff(tid)) + 1
    for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [n]])):
        ok = ~edge[lo:hi]
        if ok.sum() >= 2:
            idx = np.flatnonzero(ok)
            res = smooth_track(t[lo:hi][idx], fx[lo:hi][idx], fy[lo:hi][idx], scale, window_sec, order, fx[lo:hi][idx], y1[lo:hi][idx])
            # Border rows take the kinematics of the nearest interior row.
            nearest = idx[np.clip(np.searchsorted(idx, np.arange(hi - lo)), 0, len(idx) - 1)]
            prev = idx[np.clip(np.searchsorted(idx, np.arange(hi - lo)) - 1, 0, len(idx) - 1)]
            pick = np.where(np.abs(prev - np.arange(hi - lo)) < np.abs(nearest - np.arange(hi - lo)), prev, nearest)
            pos = np.searchsorted(idx, pick)
            for k, arr in zip(names, res):
                cols[k][lo:hi] = arr[pos]
            cols["sx"][lo:hi][~ok] = fx[lo:hi][~ok]
            cols["sy"][lo:hi][~ok] = fy[lo:hi][~ok]
        else:
            res = smooth_track(t[lo:hi], fx[lo:hi], fy[lo:hi], scale, window_sec, order, fx[lo:hi], y1[lo:hi])
            for k, arr in zip(names, res):
                cols[k][lo:hi] = arr
    for k, arr in cols.items():
        df[k] = arr
    return df
