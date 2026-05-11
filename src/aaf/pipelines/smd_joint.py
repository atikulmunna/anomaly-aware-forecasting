"""SMD joint MDN-LSTM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aaf.data.preprocessing import Standardizer, WindowedDataset
from aaf.data.smd import prepare_smd_windowed_datasets
from aaf.models.joint import JointMDNLSTMConfig
from aaf.models.joint_loss import JointLossConfig
from aaf.train.loop import TrainingConfig


@dataclass(frozen=True)
class SMDJointConfig:
    root: Path
    machine_ids: tuple[str, ...] | None = None
    validation_fraction: float = 0.2
    lookback: int = 100
    horizon: int = 1
    stride: int = 1
    n_regimes: int = 3
    hidden_size: int = 32
    num_layers: int = 1
    n_components: int = 3
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    smoothness_weight: float = 0.1
    supervised_regime_weight: float = 0.0
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
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")
        if self.supervised_regime_weight < 0.0:
            raise ValueError("supervised_regime_weight must be non-negative")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least 2")


def build_smd_joint_datasets(
    config: SMDJointConfig,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, tuple[Standardizer, ...]]:
    """Build SMD windowed datasets for joint model training."""

    config.validate()
    return prepare_smd_windowed_datasets(
        config.root,
        machine_ids=config.machine_ids,
        validation_fraction=config.validation_fraction,
        lookback=config.lookback,
        horizon=config.horizon,
        stride=config.stride,
    )


def smd_joint_model_config(
    dataset: WindowedDataset,
    config: SMDJointConfig,
) -> JointMDNLSTMConfig:
    """Create a joint model config from SMD dataset dimensions."""

    return JointMDNLSTMConfig(
        input_size=dataset.windows.shape[-1],
        output_size=dataset.targets.shape[-1],
        n_regimes=config.n_regimes,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=config.horizon,
        n_components=config.n_components,
    )


def smd_joint_loss_config(config: SMDJointConfig) -> JointLossConfig:
    """Create joint objective weights for an SMD run."""

    return JointLossConfig(
        smoothness_weight=config.smoothness_weight,
        supervised_regime_weight=config.supervised_regime_weight,
    )


def smd_joint_training_config(config: SMDJointConfig) -> TrainingConfig:
    """Create training loop parameters for an SMD joint run."""

    return TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
