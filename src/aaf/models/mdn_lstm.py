"""MDN-LSTM forecaster without regime head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aaf.models.mixture import MixtureParams


@dataclass(frozen=True)
class MDNLSTMConfig:
    input_size: int
    output_size: int
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


class MDNLSTMForecaster(nn.Module):
    """LSTM backbone with diagonal Gaussian mixture forecasting head."""

    def __init__(self, config: MDNLSTMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.backbone = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.logit_head = nn.Linear(
            config.hidden_size,
            config.horizon * config.n_components,
        )
        self.mean_head = nn.Linear(
            config.hidden_size,
            config.horizon * config.n_components * config.output_size,
        )
        self.std_head = nn.Linear(
            config.hidden_size,
            config.horizon * config.n_components * config.output_size,
        )

    def forward(self, history: Tensor) -> MixtureParams:
        """Return mixture parameters for every input timestep.

        Input shape: (B, L, input_size)
        Output shapes:
        - logits: (B, L, horizon, M)
        - means/raw_stds: (B, L, horizon, M, output_size)
        """

        if history.ndim != 3:
            raise ValueError("history must have shape (B, L, input_size)")
        if history.shape[-1] != self.config.input_size:
            raise ValueError("history feature dimension must match input_size")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")

        hidden, _ = self.backbone(history)
        batch_size, sequence_length, _ = hidden.shape
        logits = self.logit_head(hidden).reshape(
            batch_size,
            sequence_length,
            self.config.horizon,
            self.config.n_components,
        )
        means = self.mean_head(hidden).reshape(
            batch_size,
            sequence_length,
            self.config.horizon,
            self.config.n_components,
            self.config.output_size,
        )
        raw_stds = self.std_head(hidden).reshape(
            batch_size,
            sequence_length,
            self.config.horizon,
            self.config.n_components,
            self.config.output_size,
        )
        params = MixtureParams(logits=logits, means=means, raw_stds=raw_stds)
        params.validate()
        return params

    def forecast_last(self, history: Tensor) -> MixtureParams:
        """Return mixture parameters for the final history timestep only."""

        params = self(history)
        return MixtureParams(
            logits=params.logits[:, -1:, :, :],
            means=params.means[:, -1:, :, :, :],
            raw_stds=params.raw_stds[:, -1:, :, :, :],
        )
