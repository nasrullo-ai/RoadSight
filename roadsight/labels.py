"""Dev labels in the official ground-truth shape: ``{"C3896.MP4": {"duration", "fps", "events"}}``."""

from __future__ import annotations

import json
from pathlib import Path


def load_gt(path: str | Path) -> dict[str, dict]:
    """Read a ground-truth file; an older ``{"videos": {...}}`` wrapper is unwrapped. Missing file -> {}."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data.get("videos"), dict):
        data = data["videos"]
    return {k: v for k, v in data.items() if isinstance(v, dict) and "events" in v}
