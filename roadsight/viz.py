"""Annotated video rendering and timeline plots (used by tools/render.py and the web demo)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from roadsight import CLASSES

CLS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}
CLS_COLORS = {0: (60, 200, 255), 1: (255, 170, 60), 2: (80, 220, 120), 3: (255, 120, 200), 5: (240, 200, 60), 7: (200, 140, 255)}
EVENT_COLORS = {
    c: tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(180 * i / len(CLASSES)), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
    for i, c in enumerate(CLASSES)
}


def _writer(path: Path, fps: float, size: tuple[int, int]):
    """H.264 writer via imageio-ffmpeg (browser-playable); falls back to OpenCV mp4v."""
    try:
        import imageio_ffmpeg

        gen = imageio_ffmpeg.write_frames(
            str(path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=2,
            output_params=["-crf", "28", "-preset", "medium", "-movflags", "+faststart"],  # ~30 MB per 5 min at 768 px
        )
        gen.send(None)
        return lambda frame: gen.send(np.ascontiguousarray(frame[:, :, ::-1])), gen.close
    except Exception:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        return vw.write, vw.release


def render_video(
    video_path: str,
    result,
    risk: list | None,
    out_path: str | Path,
    width: int = 960,
    progress: Callable[[float], None] | None = None,
    max_seconds: float | None = None,
    every: int = 1,
) -> Path:
    """Draw tracks, scene overlay, active events and the risk score onto every ``every``-th frame
    (the output plays at ``fps / every``; skipped frames are grabbed, not decoded)."""
    every = max(1, int(every))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = result.meta
    scale = width / float(meta.width)
    height = int(round(meta.height * scale / 2)) * 2
    width = int(round(width / 2)) * 2
    df = result.tracks.df
    by_frame = {int(f): g for f, g in df.groupby("frame", sort=True)}
    overlay = _scene_overlay(result.scene, meta.width, meta.height, width, height)
    risk_arr = np.asarray(risk, dtype=np.float64) if risk else np.zeros((0, 2))
    events = result.events
    trails: dict[int, list] = {}

    cap = cv2.VideoCapture(video_path)
    fps = meta.fps
    write, close = _writer(out_path, fps / every, (width, height))
    n_total = meta.n_frames or 1
    last_boxes = None
    fresh = False  # new boxes since the last written frame
    idx = 0
    try:
        while True:
            if idx % every:
                if not cap.grab():
                    break
                g = by_frame.get(idx)
                if g is not None:
                    last_boxes, fresh = g, True
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok or (max_seconds and idx / fps > max_seconds):
                break
            t = idx / fps
            img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if overlay is not None:
                img = cv2.addWeighted(img, 1.0, overlay, 0.35, 0)
            g = by_frame.get(idx)
            if g is not None:
                last_boxes, fresh = g, True
            if last_boxes is not None and abs(float(last_boxes["t"].iloc[0]) - t) < 0.3:
                _draw_tracks(img, last_boxes, scale, trails, update=fresh)
            fresh = False
            active = [e for e in events if e[0] <= t <= e[1]]
            _draw_banner(img, t, active, risk_arr)
            write(img)
            idx += 1
            if progress and idx % 25 == 0:
                progress(min(1.0, idx / n_total))
    finally:
        cap.release()
        close()
    return out_path


def _scene_overlay(scene, w0, h0, w, h) -> np.ndarray | None:
    ov = np.zeros((h, w, 3), np.uint8)
    sx, sy = w / w0, h / h0
    drawn = False
    if scene.carriageway:
        for p in scene.carriageway:
            cv2.fillPoly(ov, [np.round(p * [sx, sy]).astype(np.int32)], (90, 60, 20))
        drawn = True
    elif scene.auto is not None and scene.auto.road.any():
        m = cv2.resize(scene.auto.road.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov[m] = (90, 60, 20)
        drawn = True
    for p in scene.crosswalks:
        cv2.fillPoly(ov, [np.round(p * [sx, sy]).astype(np.int32)], (160, 160, 160))
    if scene.auto is not None and scene.auto.crosswalk is not None and not scene.crosswalks:
        m = cv2.resize(scene.auto.crosswalk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov[m] = (160, 160, 160)
    for line in scene.stop_lines:
        cv2.polylines(ov, [np.round(line.points * [sx, sy]).astype(np.int32)], False, (0, 0, 255), 3)
        drawn = True
    for line in scene.solid_lines:
        cv2.polylines(ov, [np.round(line * [sx, sy]).astype(np.int32)], False, (0, 255, 255), 2)
        drawn = True
    for z in scene.no_u_turn_zones:
        cv2.polylines(ov, [np.round(z * [sx, sy]).astype(np.int32)], True, (0, 120, 255), 2)
    return ov if drawn else None


def _draw_tracks(img, g, scale, trails, update: bool = True) -> None:
    for row in g.itertuples(index=False):
        c = CLS_COLORS.get(int(row.cls), (200, 200, 200))
        x1, y1, x2, y2 = (int(v * scale) for v in (row.x1, row.y1, row.x2, row.y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"{CLS_NAMES.get(int(row.cls), int(row.cls))} #{int(row.track_id)} {row.speed:.1f}"
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        tr = trails.setdefault(int(row.track_id), [])
        if update:
            tr.append((int(row.sx * scale), int(row.sy * scale)))
            del tr[:-25]
        if len(tr) > 1:
            cv2.polylines(img, [np.array(tr, np.int32)], False, c, 1, cv2.LINE_AA)


def _draw_banner(img, t, active, risk_arr) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(img, f"t={t:6.2f}s", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    x = 110
    for _, _, lab in active:
        c = EVENT_COLORS.get(lab, (0, 0, 255))
        txt = lab.replace("_", " ").upper()
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x - 4, 4), (x + tw + 4, 26), c, -1)
        cv2.putText(img, txt, (x, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        x += tw + 14
    if len(risk_arr):
        k = int(np.clip(np.searchsorted(risk_arr[:, 0], t, side="right") - 1, 0, len(risk_arr) - 1))
        r = float(risk_arr[k, 1])
        bw = 160
        cv2.rectangle(img, (w - bw - 70, 8), (w - 70, 22), (60, 60, 60), -1)
        col = (60, 200, 60) if r < 0.3 else (0, 200, 255) if r < 0.5 else (0, 0, 255)
        cv2.rectangle(img, (w - bw - 70, 8), (w - bw - 70 + int(bw * r), 22), col, -1)
        cv2.putText(img, f"risk {r:.2f}", (w - 64, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def timeline_png(events: list, risk: list | None, duration: float, out_path: str | Path, gt: list | None = None) -> Path:
    """Event timeline (one row per class, predictions vs ground truth) above the risk curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({e[2] for e in events} | {e[2] for e in (gt or [])}) or ["(no events)"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 1.2 + 0.35 * len(labels) + 1.6), sharex=True, gridspec_kw={"height_ratios": [max(1, len(labels)), 2]}
    )
    for i, lab in enumerate(labels):
        for s, e, l2 in gt or []:
            if l2 == lab:
                ax.barh(i + 0.2, e - s, left=s, height=0.35, color="#9aa4b1")
        for s, e, l2 in events:
            if l2 == lab:
                ax.barh(i - 0.2, e - s, left=s, height=0.35, color="#2f6fdf")
    ax.set_yticks(range(len(labels)), [x.replace("_", " ") for x in labels])
    ax.set_xlim(0, max(duration, 1))
    ax.set_title("events (blue = predicted, grey = ground truth)", fontsize=9, loc="left")
    if risk:
        r = np.asarray(risk)
        ax2.plot(r[:, 0], r[:, 1], color="#d9534f", lw=1)
        ax2.axhline(0.5, color="#888", lw=0.8, ls="--")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("risk")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
