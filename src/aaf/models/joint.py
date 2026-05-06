"""Joint MDN forecasting and regime-detection model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from aaf.models.mixture import MixtureParams


@dataclass(frozen=True)
class JointMDNLSTMConfig:
    input_size: int
    output_size: int
    n_regimes: int
    hidden_size: int = 128
    num_layers: int = 2
    horizon: int = 1
    n_components: int = 3
    dropout: float = 0.0

    def validate(self) -> None:
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        if self.output_size < 1:
            raise ValueError("output_size must be positive")
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class JointOutput:
    """Joint model output for forecasting and regime inference."""

    forecast: MixtureParams
    regime_logits: Tensor

    def validate(self) -> None:
        self.forecast.validate()
        if self.regime_logits.ndim != 3:
            raise ValueError("regime_logits must have shape (B, T, K)")
        if self.regime_logits.shape[:2] != self.forecast.logits.shape[:2]:
            raise ValueError("regime_logits must share batch and time dimensions with forecast")
        if not torch.isfinite(self.regime_logits).all():
            raise ValueError("regime_logits must be finite")

    @property
    def regime_probs(self) -> Tensor:
        return torch.softmax(self.regime_logits, dim=-1)
