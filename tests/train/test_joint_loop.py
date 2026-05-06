import numpy as np

from aaf.eval.forecasting import MixtureForecast
from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster
from aaf.train.joint_loop import JointPrediction, JointTrainingResult
from aaf.train.loop import TrainingHistory


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
