"""Config loading. All relative paths in configs resolve against the repo root."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def deep_update(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path = "configs/default.yaml", overrides: dict | None = None) -> dict[str, Any]:
    """Load a YAML config, merge its ``overrides_file`` (written by tools/tune.py) if present, then ``overrides``."""
    with open(resolve_path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extra = cfg.get("overrides_file")
    if extra and resolve_path(extra).exists():
        with open(resolve_path(extra), encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg
