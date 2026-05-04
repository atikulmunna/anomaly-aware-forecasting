import numpy as np
import pytest

from aaf.data.preprocessing import WindowedDataset
from aaf.models.mdn_lstm import MDNLSTMConfig
from aaf.train.loop import (
    TrainingConfig,
    evaluate_mdn_lstm_loss,
    predict_mdn_lstm,
    train_mdn_lstm,
)


def make_linear_dataset(
    *,
    n_examples: int = 64,
    lookback: int = 6,
    horizon: int = 1,
) -> WindowedDataset:
    base = np.linspace(-1.0, 1.0, n_examples + lookback + horizon + 1, dtype=np.float64)
    windows = np.stack([base[idx : idx + lookback] for idx in range(n_examples)], axis=0)
    targets = np.stack(
        [base[idx + lookback : idx + lookback + horizon] for idx in range(n_examples)],
        axis=0,
    )
    return WindowedDataset(
        windows=windows[:, :, np.newaxis],
        targets=targets[:, :, np.newaxis],
        regime_labels=np.zeros(n_examples, dtype=np.int64),
        anomaly_labels=np.zeros(n_examples, dtype=np.int64),
    )


def test_train_mdn_lstm_reduces_loss_on_tiny_sequence() -> None:
    dataset = make_linear_dataset()
    model_config = MDNLSTMConfig(
        input_size=1,
        output_size=1,
        hidden_size=12,
        num_layers=1,
        horizon=1,
        n_components=2,
    )
    result = train_mdn_lstm(
        dataset,
        model_config,
        TrainingConfig(epochs=8, batch_size=16, learning_rate=0.03, seed=7),
        validation_dataset=dataset,
    )

    assert result.history.train_loss[-1] < result.history.train_loss[0]
    assert result.history.validation_loss[-1] < result.history.validation_loss[0]


def test_evaluate_mdn_lstm_loss_returns_finite_value() -> None:
    dataset = make_linear_dataset(n_examples=16)
    result = train_mdn_lstm(
        dataset,
        MDNLSTMConfig(input_size=1, output_size=1, hidden_size=6, num_layers=1),
        TrainingConfig(epochs=1, batch_size=8, learning_rate=0.01, seed=3),
    )

    loss = evaluate_mdn_lstm_loss(result.model, dataset)

    assert np.isfinite(loss)


def test_predict_mdn_lstm_returns_numpy_mixture_forecast() -> None:
    dataset = make_linear_dataset(n_examples=12, horizon=2)
    result = train_mdn_lstm(
        dataset,
        MDNLSTMConfig(
            input_size=1,
            output_size=1,
            hidden_size=6,
            num_layers=1,
            horizon=2,
            n_components=3,
        ),
        TrainingConfig(epochs=1, batch_size=4, learning_rate=0.01, seed=5),
    )

    forecast = predict_mdn_lstm(result.model, dataset, batch_size=5)

    assert forecast.weights.shape == (12, 2, 3)
    assert forecast.means.shape == (12, 2, 3, 1)
    assert forecast.stds.shape == (12, 2, 3, 1)
    assert np.allclose(forecast.weights.sum(axis=-1), 1.0)


def test_training_rejects_dataset_model_horizon_mismatch() -> None:
    dataset = make_linear_dataset(horizon=2)

    with pytest.raises(ValueError, match="horizon"):
        train_mdn_lstm(
            dataset,
            MDNLSTMConfig(input_size=1, output_size=1, horizon=1),
            TrainingConfig(epochs=1),
        )


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0).validate()
