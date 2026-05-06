"""Joint loss terms for forecasting and regime detection."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from aaf.models.joint import JointOutput
from aaf.models.mixture import mixture_nll


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


def forecast_nll_loss(target: Tensor, output: JointOutput) -> Tensor:
    """Return forecast NLL for a joint output."""

    return mixture_nll(target, output.forecast)


def regime_smoothness_loss(regime_logits: Tensor) -> Tensor:
    """Return mean KL(q_t || q_{t-1}) across adjacent timesteps."""

    if regime_logits.ndim != 3:
        raise ValueError("regime_logits must have shape (B, T, K)")
    if regime_logits.shape[1] < 2:
        return regime_logits.sum() * 0.0
    log_probs = F.log_softmax(regime_logits, dim=-1)
    probs = log_probs.exp()
    return F.kl_div(
        log_probs[:, :-1, :],
        probs[:, 1:, :],
        reduction="batchmean",
        log_target=False,
    )


def supervised_regime_loss(regime_logits: Tensor, regime_labels: Tensor) -> Tensor:
    """Return cross-entropy for supervised synthetic regime labels."""

    if regime_logits.ndim != 3:
        raise ValueError("regime_logits must have shape (B, T, K)")
    if regime_labels.shape != regime_logits.shape[:2]:
        raise ValueError("regime_labels must have shape (B, T)")
    return F.cross_entropy(
        regime_logits.reshape(-1, regime_logits.shape[-1]),
        regime_labels.reshape(-1).long(),
    )
