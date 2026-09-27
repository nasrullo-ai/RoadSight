"""Assemble the website as a Hugging Face *Static* Space (free tier) in build/static-site/.

All pages, renders and EDA are served as static files. The live demo needs the FastAPI backend (web/app.py)
running elsewhere; set its public URL in data/api.json (``{"api": "https://..."}``) and the Demo page uses it,
or tells visitors the demo is offline when it cannot reach it.

    python tools/make_static_site.py [--api https://example.trycloudflare.com]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "static-site"

README = """---
title: RoadSight
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
---

RoadSight: traffic event detection and causal accident risk for a fixed CCTV camera (WIUT Hackathon 2026).
Code: https://github.com/nasrullo-ai/RoadSight
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="", help="public URL of the demo backend (empty: demo shows as offline)")
    args = ap.parse_args(argv)
    for need in ("media/index.json", "eda/eda.json"):
        if not (ROOT / "web" / "static" / need).exists():
            print(f"missing web/static/{need}: run `make render eda` first", file=sys.stderr)
            return 1
    if OUT.exists():  # keep an existing git clone of the Space (.git), replace everything else
        for p in OUT.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copytree(ROOT / "web" / "static", OUT, dirs_exist_ok=True)
    shutil.copy2(ROOT / "predictions_samples.json", OUT / "data" / "predictions_samples.json")
    (OUT / "data" / "api.json").write_text(json.dumps({"api": args.api}, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitattributes").write_text(
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Static site assembled in {OUT} ({size:.1f} MB); demo backend: {args.api or 'none (offline notice)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
