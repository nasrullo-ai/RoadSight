"""Assemble a Hugging Face Docker Space (website + CPU live demo) in build/space/.

    python tools/make_space.py
    cd build/space && git init && git remote add origin https://huggingface.co/spaces/<user>/roadsight
    git add . && git commit -m "RoadSight demo" && git push -u origin main

Only the small detector is shipped (the Space runs on CPU). Rendered sample videos over 10 MB
need Git LFS on the Space (`git lfs track "*.mp4"`).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"

DOCKERFILE = """FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
COPY --chown=user . .
USER user
ENV YOLO_OFFLINE=1 HF_HUB_OFFLINE=1 YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/mpl ROADSIGHT_JOBS=/tmp/roadsight-jobs \\
    ROADSIGHT_CONFIG=configs/default.yaml
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1
torchvision==0.20.1
ultralytics==8.3.40
opencv-python==4.10.0.84
numpy==2.1.3
pandas==2.2.3
scipy==1.14.1
PyYAML==6.0.3
lap==0.5.13
matplotlib==3.9.4
pillow==11.1.0
fastapi==0.141.1
uvicorn==0.54.0
python-multipart==0.0.32
imageio-ffmpeg==0.6.0
"""

README = """---
title: RoadSight
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

RoadSight website and live demo: traffic event detection and causal accident risk for fixed CCTV video.
Source: see the Links page.
"""


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "roadsight", OUT / "roadsight", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "web", OUT / "web", ignore=shutil.ignore_patterns("__pycache__", "jobs"))
    (OUT / "configs").mkdir()
    for f in ("default.yaml", "scene.json", "tuned.yaml"):
        if (ROOT / "configs" / f).exists():
            shutil.copy2(ROOT / "configs" / f, OUT / "configs" / f)
    (OUT / "weights").mkdir()
    for f in ("yolo11n.pt", "risk.json"):
        shutil.copy2(ROOT / "weights" / f, OUT / "weights" / f)
    shutil.copy2(ROOT / "solution.py", OUT / "solution.py")
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUT / "requirements-space.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitattributes").write_text(
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Space assembled in {OUT} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
