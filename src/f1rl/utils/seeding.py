"""One seeding utility for the whole project (TECHNICAL_DESIGN.md §14).

Seeds Python and NumPy together; PyTorch is seeded too when it is importable (it is a
training-time dependency, not needed by the Phase 1 server). Record the returned seed
with every run.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python ``random``, NumPy, ``PYTHONHASHSEED``, and PyTorch if available.

    Returns the seed so callers can log it alongside the run.
    """
    random.seed(seed)
    # Seed the legacy global RNG on purpose: callers (and libraries) use np.random.* .
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # torch is optional at simulation time.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass

    return seed
