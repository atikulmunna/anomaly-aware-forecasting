"""Training loop for joint MDN-LSTM regime models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from aaf.data.preprocessing import WindowedDataset
from aaf.data.synthetic import FloatArray, IntArray
from aaf.eval.forecasting import MixtureForecast
from aaf.models.joint import JointMDNLSTMForecaster
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
