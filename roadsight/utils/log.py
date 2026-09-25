"""Logging to stderr only, so stdout stays clean for the harness."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str = "roadsight") -> logging.Logger:
    """Return a logger writing to stderr; level from ROADSIGHT_LOG (default INFO)."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s", "%H:%M:%S"))
        root = logging.getLogger("roadsight")
        root.addHandler(handler)
        root.setLevel(os.environ.get("ROADSIGHT_LOG", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
