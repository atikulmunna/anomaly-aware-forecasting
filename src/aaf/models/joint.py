"""Joint MDN forecasting and regime-detection model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from aaf.models.mixture import MixtureParams


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
