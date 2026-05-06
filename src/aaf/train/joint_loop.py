"""Training loop for joint MDN-LSTM regime models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from aaf.data.preprocessing import WindowedDataset
from aaf.data.synthetic import FloatArray, IntArray
from aaf.eval.forecasting import MixtureForecast
from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster
from aaf.train.loop import TrainingHistory


@dataclass(frozen=True)
class JointTrainingResult:
    model: JointMDNLSTMForecaster
    history: TrainingHistory


@dataclass(frozen=True)
class JointPrediction:
    forecast: MixtureForecast
    regime_probs: FloatArray
    regime_labels: IntArray


def to_joint_tensor_dataset(dataset: WindowedDataset) -> TensorDataset:
    """Convert a windowed dataset to tensors including regime labels."""

    return TensorDataset(
        torch.as_tensor(dataset.windows, dtype=torch.float32),
        torch.as_tensor(dataset.targets, dtype=torch.float32),
        torch.as_tensor(dataset.regime_labels, dtype=torch.long),
    )


def validate_joint_dataset_matches_model(
    dataset: WindowedDataset,
    model_config: JointMDNLSTMConfig,
) -> None:
    """Validate windowed data against joint model dimensions."""

    if dataset.windows.shape[-1] != model_config.input_size:
        raise ValueError("dataset window channel count must match model input_size")
    if dataset.targets.shape[-1] != model_config.output_size:
        raise ValueError("dataset target channel count must match model output_size")
    if dataset.targets.shape[1] != model_config.horizon:
        raise ValueError("dataset target horizon must match model horizon")
    if np.max(dataset.regime_labels) >= model_config.n_regimes:
        raise ValueError("dataset regime labels must be less than n_regimes")
