"""End-to-end synthetic joint MDN-LSTM training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointSyntheticConfig:
    seed: int = 0
    n_train_configs: int = 4
    n_validation_configs: int = 1
    n_test_configs: int = 1
    series_length: int = 500
    burn_in: int = 50
    lookback: int = 32
    horizon: int = 1
    stride: int = 1
    n_regimes: int = 3
    n_channels: int = 1
    ar_order: int = 2
    hidden_size: int = 32
    num_layers: int = 1
    n_components: int = 3
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    smoothness_weight: float = 0.1
    supervised_regime_weight: float = 0.0
    energy_samples: int = 64

    def validate(self) -> None:
        if self.series_length < self.lookback + self.horizon + 1:
            raise ValueError("series_length must exceed lookback + horizon")
        if self.n_train_configs < 1:
            raise ValueError("n_train_configs must be positive")
        if self.n_validation_configs < 1:
            raise ValueError("n_validation_configs must be positive")
        if self.n_test_configs < 1:
            raise ValueError("n_test_configs must be positive")
        if self.smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")
        if self.supervised_regime_weight < 0.0:
            raise ValueError("supervised_regime_weight must be non-negative")
