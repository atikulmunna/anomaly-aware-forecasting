"""SMD seasonal-naive baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SMDBaselineConfig:
    root: Path
    machine_ids: tuple[str, ...] | None = None
    validation_fraction: float = 0.2
    lookback: int = 100
    horizon: int = 1
    stride: int = 1
    season_length: int = 1
    energy_samples: int = 128
    seed: int = 0

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.season_length < 1:
            raise ValueError("season_length must be positive")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least 2")
