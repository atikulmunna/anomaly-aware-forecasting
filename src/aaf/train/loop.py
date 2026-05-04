"""Training loop for MDN-LSTM forecasters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from aaf.data.preprocessing import WindowedDataset
from aaf.eval.forecasting import MixtureForecast
from aaf.models.mdn_lstm import MDNLSTMConfig, MDNLSTMForecaster
from aaf.models.mixture import mixture_nll
from aaf.train.seed import seed_everything


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided")


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1]

    @property
    def final_validation_loss(self) -> float | None:
        if not self.validation_loss:
            return None
        return self.validation_loss[-1]


@dataclass(frozen=True)
class TrainingResult:
    model: MDNLSTMForecaster
    history: TrainingHistory


def train_mdn_lstm(
    train_dataset: WindowedDataset,
    model_config: MDNLSTMConfig,
    training_config: TrainingConfig,
    *,
    validation_dataset: WindowedDataset | None = None,
) -> TrainingResult:
    """Train an MDN-LSTM forecaster on windowed data."""

    training_config.validate()
    model_config.validate()
    _validate_dataset_matches_model(train_dataset, model_config)
    if validation_dataset is not None:
        _validate_dataset_matches_model(validation_dataset, model_config)

    seed_everything(training_config.seed)
    device = torch.device(training_config.device)
    model = MDNLSTMForecaster(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loader = DataLoader(
        _to_tensor_dataset(train_dataset),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=_torch_generator(training_config.seed),
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(training_config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for windows, targets in loader:
            windows = windows.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, windows, targets)
            loss.backward()  # type: ignore[no-untyped-call]
            if training_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_losses.append(float(np.mean(epoch_losses)))
        if validation_dataset is not None:
            validation_losses.append(
                evaluate_mdn_lstm_loss(model, validation_dataset, device=device)
            )

    return TrainingResult(
        model=model,
        history=TrainingHistory(
            train_loss=tuple(train_losses),
            validation_loss=tuple(validation_losses),
        ),
    )


def evaluate_mdn_lstm_loss(
    model: MDNLSTMForecaster,
    dataset: WindowedDataset,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate mean MDN NLL on a windowed dataset."""

    model.eval()
    resolved_device = torch.device(device)
    losses: list[float] = []
    with torch.no_grad():
        for windows, targets in DataLoader(_to_tensor_dataset(dataset), batch_size=batch_size):
            loss = _batch_loss(model, windows.to(resolved_device), targets.to(resolved_device))
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def predict_mdn_lstm(
    model: MDNLSTMForecaster,
    dataset: WindowedDataset,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> MixtureForecast:
    """Predict a NumPy mixture forecast artifact for a windowed dataset."""

    model.eval()
    resolved_device = torch.device(device)
    logits: list[np.ndarray] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    with torch.no_grad():
        for windows, _targets in DataLoader(_to_tensor_dataset(dataset), batch_size=batch_size):
            params = model.forecast_last(windows.to(resolved_device))
            logits.append(params.logits[:, 0].cpu().numpy())
            means.append(params.means[:, 0].cpu().numpy())
            stds.append(params.stds[:, 0].cpu().numpy())

    logits_array = np.concatenate(logits, axis=0)
    weights = _softmax_numpy(logits_array, axis=-1)
    return MixtureForecast.from_arrays(
        weights=weights,
        means=np.concatenate(means, axis=0),
        stds=np.concatenate(stds, axis=0),
    )


def _batch_loss(model: MDNLSTMForecaster, windows: Tensor, targets: Tensor) -> Tensor:
    params = model.forecast_last(windows)
    target = targets.unsqueeze(1)
    return mixture_nll(target, params)


def _to_tensor_dataset(dataset: WindowedDataset) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(dataset.windows, dtype=torch.float32),
        torch.as_tensor(dataset.targets, dtype=torch.float32),
    )


def _validate_dataset_matches_model(dataset: WindowedDataset, model_config: MDNLSTMConfig) -> None:
    if dataset.windows.shape[-1] != model_config.input_size:
        raise ValueError("dataset window channel count must match model input_size")
    if dataset.targets.shape[-1] != model_config.output_size:
        raise ValueError("dataset target channel count must match model output_size")
    if dataset.targets.shape[1] != model_config.horizon:
        raise ValueError("dataset target horizon must match model horizon")


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _softmax_numpy(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return np.asarray(exp_values / np.sum(exp_values, axis=axis, keepdims=True))
