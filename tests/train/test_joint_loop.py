import numpy as np
import pytest

from aaf.eval.forecasting import MixtureForecast
from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster
from aaf.models.joint_loss import JointLossConfig
from aaf.train.joint_loop import (
    JointPrediction,
    JointTrainingResult,
    joint_batch_loss,
    to_joint_tensor_dataset,
    train_joint_mdn_lstm,
    validate_joint_dataset_matches_model,
)
from aaf.train.loop import TrainingConfig, TrainingHistory
from tests.train.test_loop import make_linear_dataset


def test_joint_training_result_holds_model_and_history() -> None:
    model = JointMDNLSTMForecaster(JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2))
    result = JointTrainingResult(
        model=model,
        history=TrainingHistory(train_loss=(1.0,), validation_loss=(1.2,)),
    )

    assert result.model is model
    assert result.history.final_train_loss == 1.0


def test_joint_prediction_holds_forecast_and_regime_arrays() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 1)),
        means=np.zeros((2, 1, 1)),
        stds=np.ones((2, 1, 1)),
    )
    prediction = JointPrediction(
        forecast=forecast,
        regime_probs=np.array([[0.7, 0.3], [0.2, 0.8]]),
        regime_labels=np.array([0, 1]),
    )

    assert prediction.regime_labels.tolist() == [0, 1]


def test_to_joint_tensor_dataset_includes_regime_labels() -> None:
    dataset = make_linear_dataset(n_examples=4)
    tensor_dataset = to_joint_tensor_dataset(dataset)
    windows, targets, regime_labels = tensor_dataset[0]

    assert tuple(windows.shape) == (6, 1)
    assert tuple(targets.shape) == (1, 1)
    assert regime_labels.item() == 0


def test_validate_joint_dataset_matches_model_accepts_compatible_data() -> None:
    validate_joint_dataset_matches_model(
        make_linear_dataset(),
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2),
    )


def test_validate_joint_dataset_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        validate_joint_dataset_matches_model(
            make_linear_dataset(horizon=2),
            JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, horizon=1),
        )


def test_joint_batch_loss_returns_finite_scalar() -> None:
    dataset = make_linear_dataset(n_examples=4)
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, hidden_size=6, num_layers=1)
    )
    windows, targets, regime_labels = to_joint_tensor_dataset(dataset).tensors

    loss = joint_batch_loss(
        model,
        windows,
        targets,
        regime_labels,
        loss_config=JointLossConfig(),
    )

    assert loss.ndim == 0
    assert np.isfinite(loss.item())


def test_train_joint_mdn_lstm_reduces_loss_on_tiny_sequence() -> None:
    dataset = make_linear_dataset(n_examples=32)
    result = train_joint_mdn_lstm(
        dataset,
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, hidden_size=8, num_layers=1),
        TrainingConfig(epochs=4, batch_size=8, learning_rate=0.03, seed=13),
        JointLossConfig(smoothness_weight=0.01, supervised_regime_weight=0.1),
        validation_dataset=dataset,
    )

    assert result.history.train_loss[-1] < result.history.train_loss[0]
