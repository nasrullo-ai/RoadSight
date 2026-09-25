"""Exploratory data analysis of the sample videos for the website EDA page.

Per video: resolution / fps / duration / lighting, object counts over time per class, motion
heatmap, trajectories coloured by direction, lane-flow field and speed distribution.

    python tools/eda.py --videos data/samples --out web/static/eda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from roadsight.pipeline import EventPipeline  # noqa: E402
from roadsight.viz import CLS_NAMES  # noqa: E402


def lighting(path: str, n: int = 12) -> dict:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vals, frame0 = [], None
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / n))
        ok, f = cap.read()
        if not ok:
            continue
        if frame0 is None:
            frame0 = f
        vals.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    m = float(np.mean(vals)) if vals else 0.0
    return {"brightness": round(m, 1), "condition": "night" if m < 60 else "dusk/overcast" if m < 100 else "day"}, frame0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/samples")
    ap.add_argument("--out", default="web/static/eda")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = EventPipeline.from_config(args.config)
    report = {"videos": []}
    for path in sorted(Path(args.videos).glob("*.mp4")):
        res = pipe.run_full(str(path))
        df = res.tracks.df
        meta = res.meta
        light, frame0 = lighting(str(path))
        stem = path.stem
        entry = {"video": path.name, **meta.to_dict(), **light}
        entry["tracks_per_class"] = (
            {CLS_NAMES.get(int(k), str(k)): int(v) for k, v in df.groupby("cls")["track_id"].nunique().items()} if len(df) else {}
        )

        # Object counts over time (1 s bins).
        counts = {}
        if len(df):
            b = (df["t"] // 1.0).astype(int)
            for cls, g in df.groupby("cls"):
                c = g.groupby(b[g.index])["track_id"].nunique()
                counts[CLS_NAMES.get(int(cls), str(cls))] = [[int(k), int(v)] for k, v in c.items()]
        entry["counts_over_time"] = counts
        mv = df[df["speed"] > 0.3]
        entry["speed_bl_s"] = {
            "p50": round(float(mv["speed"].median()), 2) if len(mv) else 0,
            "p90": round(float(mv["speed"].quantile(0.9)), 2) if len(mv) else 0,
        }
        entry["speed_hist"] = np.histogram(mv["speed"].clip(0, 8), bins=16, range=(0, 8))[0].tolist() if len(mv) else []
        entry["events"] = res.events

        bg = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB) if frame0 is not None else np.zeros((meta.height, meta.width, 3), np.uint8)
        # Motion heatmap.
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(bg, alpha=0.55)
        if len(df):
            h, xe, ye = np.histogram2d(df["foot_x"], df["foot_y"], bins=[64, 36], range=[[0, meta.width], [0, meta.height]])
            ax.imshow(np.log1p(h.T), extent=[0, meta.width, meta.height, 0], cmap="inferno", alpha=0.6)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(out / f"{stem}_heatmap.png", dpi=100)
        plt.close(fig)

        # Trajectories coloured by direction.
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(bg, alpha=0.5)
        for _, g in df.groupby("track_id"):
            if len(g) < 5:
                continue
            ang = np.degrees(np.arctan2(g["sy"].iloc[-1] - g["sy"].iloc[0], g["sx"].iloc[-1] - g["sx"].iloc[0]))
            ax.plot(g["sx"], g["sy"], color=plt.cm.hsv(((ang + 180) % 360) / 360), lw=1, alpha=0.8)
        ax.set_xlim(0, meta.width)
        ax.set_ylim(meta.height, 0)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(out / f"{stem}_trajectories.png", dpi=100)
        plt.close(fig)

        # Lane-flow field from the automatic scene.
        auto = res.scene.auto
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(bg, alpha=0.5)
        if auto is not None:
            yy, xx = np.mgrid[0 : auto.grid.gh, 0 : auto.grid.gw]
            cx = (xx.ravel() + 0.5) * auto.grid.cell
            cy = (yy.ravel() + 0.5) * auto.grid.cell
            d, s = auto.flow.direction(cx, cy)
            m = s > 0.3
            ax.quiver(cx[m], cy[m], d[m, 0], -d[m, 1], s[m], cmap="viridis", scale=40, width=0.003)
            if auto.road.any():
                road = cv2.resize(auto.road.astype(np.uint8), (meta.width, meta.height), interpolation=cv2.INTER_NEAREST)
                ax.contour(road, levels=[0.5], colors="cyan", linewidths=1)
            entry["road_fraction"] = round(float(auto.road.mean()), 3)
        ax.set_xlim(0, meta.width)
        ax.set_ylim(meta.height, 0)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(out / f"{stem}_flow.png", dpi=100)
        plt.close(fig)

        entry["images"] = {k: f"{stem}_{k}.png" for k in ("heatmap", "trajectories", "flow")}
        report["videos"].append(entry)
        print(f"[eda] {path.name}: {len(df)} rows, {df['track_id'].nunique() if len(df) else 0} tracks", file=sys.stderr)
    (out / "eda.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
