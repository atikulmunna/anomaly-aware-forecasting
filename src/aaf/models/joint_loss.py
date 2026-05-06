"""Joint loss terms for forecasting and regime detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLossConfig:
    smoothness_weight: float = 0.1
    supervised_regime_weight: float = 0.0

    def validate(self) -> None:
        if self.smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")
        if self.supervised_regime_weight < 0.0:
            raise ValueError("supervised_regime_weight must be non-negative")


@dataclass(frozen=True)
class JointLossComponents:
    total: float
    forecast_nll: float
    smoothness: float
    supervised_regime: float
