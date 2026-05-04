"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic_torch: bool
    python_hash_seed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "deterministic_torch": self.deterministic_torch,
            "python_hash_seed": self.python_hash_seed,
        }


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Seed Python, NumPy, and torch when available."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_module = _try_import_torch()
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if hasattr(torch_module, "cuda"):
            torch_module.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch_module.use_deterministic_algorithms(True, warn_only=True)
            torch_module.backends.cudnn.benchmark = False

    return SeedState(
        seed=seed,
        deterministic_torch=deterministic_torch,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
    )


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
