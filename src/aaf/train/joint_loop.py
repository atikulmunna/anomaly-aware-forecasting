"""Training loop for joint MDN-LSTM regime models."""

from __future__ import annotations

from dataclasses import dataclass

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
