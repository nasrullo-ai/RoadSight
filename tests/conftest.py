import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from synth import write_video  # noqa: E402

SAMPLES = sorted(p for p in (ROOT / "data" / "samples").glob("*") if p.suffix.lower() == ".mp4")


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory) -> Path:
    """Two 10 s clips: cut from data/samples when available, synthetic otherwise."""
    d = tmp_path_factory.mktemp("clips")
    srcs = (SAMPLES + [None, None])[:2]
    for i, src in enumerate(srcs):
        write_video(d / f"clip_{i}.mp4", seconds=10.0, size=(960, 540), source=src)
    return d


@pytest.fixture(scope="session")
def clip(clip_dir) -> Path:
    return sorted(clip_dir.glob("*.mp4"))[0]
