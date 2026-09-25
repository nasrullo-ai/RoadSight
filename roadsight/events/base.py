"""EventRule interface and the rule registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from roadsight.io.video import VideoMeta
    from roadsight.perception.signal import SignalTimeline
    from roadsight.perception.tracktable import TrackTable
    from roadsight.scene.scene import Scene


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float = 1.0
    info: dict = field(default_factory=dict)

    def to_list(self) -> list:
        return [self.start, self.end, self.label]


RULES: dict[str, type[EventRule]] = {}


def register(cls: type[EventRule]) -> type[EventRule]:
    RULES[cls.label] = cls
    return cls


class EventRule(ABC):
    """One detector per event class. ``detect`` must be a pure function of its inputs."""

    label: str = ""

    def __init__(self, cfg: dict, scene: Scene) -> None:
        self.cfg = cfg
        self.scene = scene

    def p(self, key: str, default):
        """Config parameter with a default."""
        return self.cfg.get(key, default)

    @abstractmethod
    def detect(self, tracks: TrackTable, signal: SignalTimeline, meta: VideoMeta) -> list[Segment]: ...


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j] (inclusive) of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def runs_min_duration(mask: np.ndarray, t: np.ndarray, min_sec: float, max_gap_sec: float = 0.0) -> list[tuple[int, int]]:
    """Runs of True lasting at least ``min_sec``; runs separated by <= ``max_gap_sec`` are joined."""
    rs = runs(mask)
    if not rs:
        return []
    merged = [list(rs[0])]
    for s, e in rs[1:]:
        if t[s] - t[merged[-1][1]] <= max_gap_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if t[e] - t[s] >= min_sec]
