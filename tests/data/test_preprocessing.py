import numpy as np
import pytest

from aaf.data.preprocessing import Standardizer, make_windowed_dataset, standardize_series
from aaf.data.synthetic import SyntheticSeries


def make_series() -> SyntheticSeries:
    return SyntheticSeries(
        observations=np.arange(20, dtype=np.float64).reshape(10, 2),
        regime_labels=np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 0]),
        anomaly_labels=np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
        config_id="cfg",
    )


def test_standardizer_fits_train_statistics_only() -> None:
    train = np.array([[0.0], [2.0], [4.0]])
    validation = np.array([[100.0]])

    standardizer = Standardizer.fit(train)

    assert standardizer.mean.tolist() == [2.0]
    assert standardizer.transform(validation).item() == pytest.approx(
        (100.0 - 2.0) / np.std(train, axis=0).item()
    )


def test_standardizer_inverse_transform_round_trips() -> None:
    observations = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    standardizer = Standardizer.fit(observations)

    transformed = standardizer.transform(observations)
    restored = standardizer.inverse_transform(transformed)

    assert restored == pytest.approx(observations)


def test_standardize_series_preserves_labels_and_config_id() -> None:
    series = make_series()
    standardizer = Standardizer.fit(series.observations[:5])

    standardized = standardize_series(series, standardizer)

    assert standardized.config_id == series.config_id
    assert np.array_equal(standardized.regime_labels, series.regime_labels)
    assert np.array_equal(standardized.anomaly_labels, series.anomaly_labels)
    assert not np.array_equal(standardized.observations, series.observations)


def test_make_windowed_dataset_exposes_expected_tuple_structure() -> None:
    dataset = make_windowed_dataset(make_series(), lookback=3, horizon=2, stride=2)

    assert len(dataset) == 3
    window, target, regime_label, anomaly_label = dataset[0]
    assert window.shape == (3, 2)
    assert target.shape == (2, 2)
    assert regime_label == 1
    assert anomaly_label == 1


def test_make_windowed_dataset_uses_target_window_for_anomaly_label() -> None:
    dataset = make_windowed_dataset(make_series(), lookback=4, horizon=2)

    assert dataset.anomaly_labels.tolist()[:3] == [0, 1, 1]


def test_make_windowed_dataset_rejects_too_short_series() -> None:
    with pytest.raises(ValueError, match="too short"):
        make_windowed_dataset(make_series(), lookback=9, horizon=2)
