"""Per-stage wall-clock timing, logged to stderr."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from roadsight.utils.log import get_logger

log = get_logger("roadsight.timing")


class StageTimer:
    """Accumulates seconds per named stage."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += seconds

    def summary(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.totals.items()}

    def log(self, prefix: str = "") -> None:
        parts = ", ".join(f"{k}={v:.2f}s" for k, v in self.totals.items())
        log.info("%stiming: %s", prefix, parts)
