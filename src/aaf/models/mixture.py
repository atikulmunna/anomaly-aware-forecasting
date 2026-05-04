"""PyTorch Gaussian-mixture parameter helpers and losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

MIN_STD = 1e-3


@dataclass(frozen=True)
class MixtureParams:
    """Diagonal-covariance Gaussian mixture parameters.

    Shapes:
    - logits: (B, T, H, M)
    - means: (B, T, H, M, D)
    - raw_stds: (B, T, H, M, D)
    """

    logits: Tensor
    means: Tensor
    raw_stds: Tensor

    def validate(self) -> None:
        if self.logits.ndim != 4:
            raise ValueError("logits must have shape (B, T, H, M)")
        if self.means.ndim != 5:
            raise ValueError("means must have shape (B, T, H, M, D)")
        if self.raw_stds.shape != self.means.shape:
            raise ValueError("raw_stds must have the same shape as means")
        if self.logits.shape != self.means.shape[:-1]:
            raise ValueError("logits shape must match means without channel dimension")
        if not torch.isfinite(self.logits).all():
            raise ValueError("logits must be finite")
        if not torch.isfinite(self.means).all():
            raise ValueError("means must be finite")
        if not torch.isfinite(self.raw_stds).all():
            raise ValueError("raw_stds must be finite")

    @property
    def stds(self) -> Tensor:
        return F.softplus(self.raw_stds) + MIN_STD

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.logits, dim=-1)


def mixture_nll(target: Tensor, params: MixtureParams) -> Tensor:
    """Return mean negative log-likelihood for diagonal Gaussian mixtures."""

    return mixture_nll_values(target, params).mean()


def mixture_nll_values(target: Tensor, params: MixtureParams) -> Tensor:
    """Return NLL values with shape (B, T, H)."""

    params.validate()
    if target.shape != params.means.shape[:-2] + (params.means.shape[-1],):
        raise ValueError("target must have shape (B, T, H, D)")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")

    stds = params.stds
    diff = target.unsqueeze(-2) - params.means
    log_prob = -0.5 * (
        ((diff / stds) ** 2)
        + (2.0 * torch.log(stds))
        + torch.log(torch.tensor(2.0 * torch.pi, dtype=target.dtype, device=target.device))
    )
    log_prob = log_prob.sum(dim=-1)
    log_mix = torch.logsumexp(torch.log_softmax(params.logits, dim=-1) + log_prob, dim=-1)
    return -log_mix


def mixture_mean(params: MixtureParams) -> Tensor:
    """Return predictive mean with shape (B, T, H, D)."""

    params.validate()
    return torch.sum(params.weights.unsqueeze(-1) * params.means, dim=-2)
