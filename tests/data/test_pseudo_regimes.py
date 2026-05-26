import numpy as np
import pytest

from aaf.data.preprocessing import WindowedDataset
from aaf.data.pseudo_regimes import (
    assign_pseudo_regime_labels,
    fit_pseudo_regime_model,
    nearest_centroid_labels,
    pseudo_regime_features,
    replace_regime_labels,
)


def make_dataset(values: np.ndarray) -> WindowedDataset:
    return WindowedDataset(
        windows=values.astype(np.float32),
        targets=values[:, -1:, :].astype(np.float32),
        regime_labels=np.zeros(values.shape[0], dtype=np.int64),
        anomaly_labels=np.zeros(values.shape[0], dtype=np.int64),
    )


def test_pseudo_regime_features_summarize_each_window() -> None:
    windows = np.array([[[0.0], [2.0]], [[4.0], [8.0]]])

    features = pseudo_regime_features(windows)

    assert features.shape == (2, 4)
    assert features[0].tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])


def test_fit_pseudo_regime_model_assigns_nearest_centroids() -> None:
    windows = np.array(
        [
            [[0.0], [0.1]],
            [[0.2], [0.1]],
            [[10.0], [10.1]],
            [[10.2], [10.1]],
        ]
    )

    model = fit_pseudo_regime_model(windows, n_regimes=2, seed=7)
    labels = model.predict(windows)

    assert set(labels.tolist()) == {0, 1}
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_assign_pseudo_regime_labels_preserves_targets_and_anomaly_labels() -> None:
    train = make_dataset(
        np.array(
            [
                [[0.0], [0.1]],
                [[0.2], [0.1]],
                [[10.0], [10.1]],
                [[10.2], [10.1]],
            ]
        )
    )
    validation = make_dataset(np.array([[[0.1], [0.0]], [[10.1], [10.0]]]))
    test = make_dataset(np.array([[[0.3], [0.2]], [[9.9], [10.0]]]))

    pseudo_train, pseudo_validation, pseudo_test, _model = assign_pseudo_regime_labels(
        train,
        validation,
        test,
        n_regimes=2,
        seed=7,
    )

    assert set(pseudo_train.regime_labels.tolist()) == {0, 1}
    assert pseudo_validation.regime_labels.shape == validation.regime_labels.shape
    assert pseudo_test.regime_labels.shape == test.regime_labels.shape
    assert np.array_equal(pseudo_test.targets, test.targets)
    assert np.array_equal(pseudo_test.anomaly_labels, test.anomaly_labels)


def test_replace_regime_labels_rejects_length_mismatch() -> None:
    dataset = make_dataset(np.ones((2, 3, 1)))

    with pytest.raises(ValueError, match="regime_labels"):
        replace_regime_labels(dataset, np.array([0, 1, 0]))


def test_nearest_centroid_labels_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="matching dimensions"):
        nearest_centroid_labels(np.ones((2, 3)), np.ones((2, 4)))
