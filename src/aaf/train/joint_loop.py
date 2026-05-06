"""Training loop for joint MDN-LSTM regime models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from aaf.data.preprocessing import WindowedDataset
from aaf.data.synthetic import FloatArray, IntArray
from aaf.eval.forecasting import MixtureForecast
from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster
from aaf.models.joint_loss import JointLossConfig, joint_loss
from aaf.train.loop import TrainingConfig, TrainingHistory
from aaf.train.seed import seed_everything


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


def joint_batch_loss(
    model: JointMDNLSTMForecaster,
    windows: Tensor,
    targets: Tensor,
    regime_labels: Tensor,
    loss_config: JointLossConfig,
) -> Tensor:
    """Return joint loss for one training batch."""

    output = model.forecast_last(windows)
    total, _components = joint_loss(
        targets.unsqueeze(1),
        output,
        loss_config,
        regime_labels=regime_labels.unsqueeze(1),
    )
    return total


def train_joint_mdn_lstm(
    train_dataset: WindowedDataset,
    model_config: JointMDNLSTMConfig,
    training_config: TrainingConfig,
    loss_config: JointLossConfig,
    *,
    validation_dataset: WindowedDataset | None = None,
) -> JointTrainingResult:
    """Train a joint MDN-LSTM regime model."""

    training_config.validate()
    loss_config.validate()
    model_config.validate()
    validate_joint_dataset_matches_model(train_dataset, model_config)
    if validation_dataset is not None:
        validate_joint_dataset_matches_model(validation_dataset, model_config)

    seed_everything(training_config.seed)
    device = torch.device(training_config.device)
    model = JointMDNLSTMForecaster(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loader = DataLoader(
        to_joint_tensor_dataset(train_dataset),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=_torch_generator(training_config.seed),
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(training_config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for windows, targets, regime_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = joint_batch_loss(
                model,
                windows.to(device),
                targets.to(device),
                regime_labels.to(device),
                loss_config,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            if training_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_losses.append(float(np.mean(epoch_losses)))
        if validation_dataset is not None:
            validation_losses.append(
                evaluate_joint_mdn_lstm_loss(model, validation_dataset, loss_config, device=device)
            )

    return JointTrainingResult(
        model=model,
        history=TrainingHistory(tuple(train_losses), tuple(validation_losses)),
    )


def evaluate_joint_mdn_lstm_loss(
    model: JointMDNLSTMForecaster,
    dataset: WindowedDataset,
    loss_config: JointLossConfig,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate mean joint loss on a windowed dataset."""

    model.eval()
    resolved_device = torch.device(device)
    losses: list[float] = []
    with torch.no_grad():
        for windows, targets, regime_labels in DataLoader(
            to_joint_tensor_dataset(dataset),
            batch_size=batch_size,
        ):
            loss = joint_batch_loss(
                model,
                windows.to(resolved_device),
                targets.to(resolved_device),
                regime_labels.to(resolved_device),
                loss_config,
            )
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
