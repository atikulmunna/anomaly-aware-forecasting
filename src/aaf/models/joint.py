"""Joint MDN forecasting and regime-detection model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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


class JointMDNLSTMForecaster(nn.Module):
    """MDN-LSTM forecaster with an explicit regime head."""

    def __init__(self, config: JointMDNLSTMConfig) -> None:
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
        self.regime_head = nn.Linear(config.hidden_size, config.n_regimes)
        forecast_input_size = config.hidden_size + config.n_regimes
        self.logit_head = nn.Linear(forecast_input_size, config.horizon * config.n_components)
        self.mean_head = nn.Linear(
            forecast_input_size,
            config.horizon * config.n_components * config.output_size,
        )
        self.std_head = nn.Linear(
            forecast_input_size,
            config.horizon * config.n_components * config.output_size,
        )

    def forward(self, history: Tensor, regime_logits_override: Tensor | None = None) -> JointOutput:
        """Return forecast mixture parameters and regime logits for every timestep."""

        if history.ndim != 3:
            raise ValueError("history must have shape (B, T, input_size)")
        if history.shape[-1] != self.config.input_size:
            raise ValueError("history feature dimension must match input_size")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")

        hidden, _ = self.backbone(history)
        regime_logits = self.regime_head(hidden)
        if regime_logits_override is not None:
            if regime_logits_override.shape != regime_logits.shape:
                raise ValueError("regime_logits_override must match regime logits shape")
            regime_logits = regime_logits_override
        forecast_input = torch.cat([hidden, F.softmax(regime_logits, dim=-1)], dim=-1)
        output = JointOutput(
            forecast=self._forecast_from_conditioned_hidden(forecast_input),
            regime_logits=regime_logits,
        )
        output.validate()
        return output

    def _forecast_from_conditioned_hidden(self, hidden: Tensor) -> MixtureParams:
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
        return MixtureParams(logits=logits, means=means, raw_stds=raw_stds)

    def forecast_last(
        self,
        history: Tensor,
        regime_logits_override: Tensor | None = None,
    ) -> JointOutput:
        """Return joint outputs for the final history timestep only."""

        output = self(history, regime_logits_override=regime_logits_override)
        return JointOutput(
            forecast=MixtureParams(
                logits=output.forecast.logits[:, -1:, :, :],
                means=output.forecast.means[:, -1:, :, :, :],
                raw_stds=output.forecast.raw_stds[:, -1:, :, :, :],
            ),
            regime_logits=output.regime_logits[:, -1:, :],
        )
