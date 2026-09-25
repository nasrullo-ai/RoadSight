"""Deterministic seeding for Python, NumPy, Torch and CUDA."""

from __future__ import annotations

import os
import random

import numpy as np

# Must be set before the first CUDA context for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int = 0) -> None:
    """Fix every RNG we touch and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
