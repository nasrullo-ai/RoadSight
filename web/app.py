"""Website + live demo backend (SPEC section 13).

POST /api/analyze -> {job_id};  GET /api/status/{id} -> progress;  GET /api/result/{id} -> events, risk, video url.
Runs the same pipeline as solution.py (CPU: nano model, higher stride), one job at a time.

    uvicorn web.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from queue import Queue

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "1")

from roadsight.pipeline import EventPipeline  # noqa: E402
from roadsight.risk import CausalRiskModel, run_risk  # noqa: E402
from roadsight.utils.log import get_logger  # noqa: E402
from roadsight.viz import CLS_NAMES, render_video  # noqa: E402

log = get_logger("roadsight.web")
STATIC = Path(__file__).resolve().parent / "static"
JOBS = Path(os.environ.get("ROADSIGHT_JOBS", Path(__file__).resolve().parent / "jobs"))
MAX_BYTES = int(os.environ.get("ROADSIGHT_MAX_MB", "100")) * 1024 * 1024
MAX_SECONDS = float(os.environ.get("ROADSIGHT_MAX_SECONDS", "120"))
KEEP_JOBS = 20
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="RoadSight demo")
# The static site may be hosted elsewhere (GitHub Pages, Vercel) and call this API across origins.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
jobs: OrderedDict[str, dict] = OrderedDict()
queue: Queue[str] = Queue()
_models: dict = {}
_lock = threading.Lock()


def models():
    if not _models:
        cfg = os.environ.get("ROADSIGHT_CONFIG", "configs/default.yaml")
        _models["pipe"] = EventPipeline.from_config(cfg)
        _models["risk"] = CausalRiskModel.from_config(cfg)
    return _models["pipe"], _models["risk"]


def set_job(job_id: str, **kw) -> None:
    with _lock:
        jobs[job_id].update(kw)


def process(job_id: str) -> None:
    job = jobs[job_id]
    d = JOBS / job_id
    video = str(d / job["filename"])
    try:
        set_job(job_id, state="running", stage="Loading models", progress=0.02)
        pipe, risk_model = models()
        set_job(job_id, stage="Detecting and tracking road users", progress=0.05)
        t0 = time.perf_counter()
        result = pipe.run_full(video, progress=lambda p, _s: set_job(job_id, progress=0.05 + 0.45 * p))
        t_a = time.perf_counter() - t0
        set_job(job_id, stage="Scoring accident risk frame by frame", progress=0.5)
        t1 = time.perf_counter()
        risk = run_risk(video, risk_model, progress=lambda p: set_job(job_id, progress=0.5 + 0.25 * p))
        t_b = time.perf_counter() - t1
        set_job(job_id, stage="Rendering the annotated video", progress=0.75)
        render_video(video, result, risk, d / "annotated.mp4", width=854, progress=lambda p: set_job(job_id, progress=0.75 + 0.24 * p))
        df = result.tracks.df
        counts = df.groupby("cls")["track_id"].nunique().to_dict() if len(df) else {}
        out = {
            "job_id": job_id,
            "video": job["filename"],
            "meta": result.meta.to_dict(),
            "events": result.events,
            "risk": risk,
            "tracks": {CLS_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()},
            "candidates": [[round(s.start, 2), round(s.end, 2), s.label, round(s.confidence, 2)] for s in result.segments],
            "timing": {"part_a_sec": round(t_a, 1), "part_b_sec": round(t_b, 1), "device": pipe.device, "stride": result.stride},
            "annotated_video_url": f"/api/video/{job_id}",
        }
        (d / "result.json").write_text(json.dumps(out), encoding="utf-8")
        (d / "predictions.json").write_text(
            json.dumps({"team": "roadsight", "videos": {job["filename"]: {"events": result.events, "risk": risk}}}), encoding="utf-8"
        )
        set_job(job_id, state="done", stage="Done", progress=1.0)
    except Exception:  # never leak a stack trace to the page
        log.error("job %s failed:\n%s", job_id, traceback.format_exc())
        set_job(
            job_id, state="error", stage="Failed", message="The video could not be analysed. Try another MP4 file (H.264, up to 2 minutes)."
        )


def worker() -> None:
    while True:
        job_id = queue.get()
        try:
            process(job_id)
        finally:
            queue.task_done()
            cleanup()


def cleanup() -> None:
    with _lock:
        while len(jobs) > KEEP_JOBS:
            old, info = next(iter(jobs.items()))
            if info.get("state") in ("queued", "running"):
                break
            jobs.pop(old)
            shutil.rmtree(JOBS / old, ignore_errors=True)


@app.on_event("startup")
def _start() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=models, daemon=True).start()  # warm up while the page loads


@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": bool(_models), "queued": queue.qsize()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.mp4").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Upload a video file ({', '.join(sorted(ALLOWED))}). '{ext or 'no extension'}' is not supported.")
    job_id = uuid.uuid4().hex[:12]
    d = JOBS / job_id
    d.mkdir(parents=True, exist_ok=True)
    safe = "input" + ext
    size = 0
    with open(d / safe, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB. Trim or compress it and upload again.")
            f.write(chunk)
    cap = cv2.VideoCapture(str(d / safe))
    ok = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    readable = ok and cap.read()[0]
    cap.release()
    if not readable or n <= 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "This file has no readable video frames. Export it as MP4 (H.264) and upload again.")
    duration = n / (fps if 1 <= fps <= 240 else 25.0)
    if duration > MAX_SECONDS + 0.5:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(
            400, f"The video is {duration:.0f} s long. The demo accepts up to {MAX_SECONDS:.0f} s; trim it and upload again."
        )
    with _lock:
        jobs[job_id] = {
            "state": "queued",
            "stage": "Waiting in queue",
            "progress": 0.0,
            "filename": safe,
            "original": name,
            "duration": round(duration, 2),
            "created": time.time(),
            "message": "",
        }
    queue.put(job_id)
    return {"job_id": job_id, "duration": round(duration, 2)}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job. Upload the video again.")
    ahead = sum(1 for j in jobs.values() if j["state"] == "queued" and j["created"] < job["created"])
    return {k: job[k] for k in ("state", "stage", "progress", "message", "duration")} | {"queue_position": ahead}


@app.get("/api/result/{job_id}")
def result(job_id: str):
    p = JOBS / job_id / "result.json"
    if not p.exists():
        raise HTTPException(404, "The result is not ready yet.")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/video/{job_id}")
def video(job_id: str):
    p = JOBS / job_id / "annotated.mp4"
    if not p.exists():
        raise HTTPException(404, "The annotated video is not ready yet.")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/json/{job_id}")
def download(job_id: str):
    p = JOBS / job_id / "predictions.json"
    if not p.exists():
        raise HTTPException(404, "The result is not ready yet.")
    return FileResponse(p, media_type="application/json", filename=f"roadsight-{job_id}.json")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
