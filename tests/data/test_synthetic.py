import numpy as np
import pytest

from aaf.data.synthetic import (
    GeneratorConfigSpace,
    SwitchingARConfig,
    generate_switching_ar,
    sample_config_split,
    sample_switching_ar_config,
)


def test_sample_switching_ar_config_has_valid_shapes_and_probabilities() -> None:
    space = GeneratorConfigSpace(n_regimes=3, n_channels=2, ar_order=2)

    config = sample_switching_ar_config(config_id="cfg", space=space, seed=123)

    assert config.transition_matrix.shape == (3, 3)
    assert config.ar_coefficients.shape == (3, 2, 2)
    assert config.intercepts.shape == (3, 2)
    assert config.noise_stds.shape == (3, 2)
    assert np.allclose(config.transition_matrix.sum(axis=1), 1.0)
    assert np.all(config.noise_stds > 0.0)


def test_generate_switching_ar_is_reproducible_for_seed() -> None:
    config = sample_switching_ar_config(
        config_id="cfg",
        space=GeneratorConfigSpace(n_regimes=3, n_channels=2, ar_order=2),
        seed=11,
    )

    first = generate_switching_ar(config, length=128, seed=99)
    second = generate_switching_ar(config, length=128, seed=99)

    assert np.array_equal(first.observations, second.observations)
    assert np.array_equal(first.regime_labels, second.regime_labels)
    assert np.array_equal(first.anomaly_labels, second.anomaly_labels)


def test_generate_switching_ar_emits_expected_shapes_and_labels() -> None:
    config = sample_switching_ar_config(
        config_id="cfg",
        space=GeneratorConfigSpace(n_regimes=4, n_channels=3, ar_order=1),
        seed=3,
    )

    series = generate_switching_ar(config, length=50, seed=4)

    assert series.observations.shape == (50, 3)
    assert series.regime_labels.shape == (50,)
    assert series.anomaly_labels.shape == (50,)
    assert series.config_id == "cfg"
    assert set(series.regime_labels.tolist()).issubset({0, 1, 2, 3})
    assert np.array_equal(series.anomaly_labels, np.zeros(50, dtype=np.int64))


def test_config_split_uses_disjoint_config_ids() -> None:
    split = sample_config_split(
        space=GeneratorConfigSpace(n_regimes=3, n_channels=1, ar_order=2),
        n_train=4,
        n_validation=2,
        n_test=3,
        seed=7,
    )

    assert len(split.train) == 4
    assert len(split.validation) == 2
    assert len(split.test) == 3
    assert len(split.all_config_ids()) == 9
    assert {config.config_id for config in split.train}.isdisjoint(
        {config.config_id for config in split.test}
    )
    assert {config.config_id for config in split.validation}.isdisjoint(
        {config.config_id for config in split.test}
    )


def test_config_split_is_reproducible() -> None:
    kwargs = {
        "space": GeneratorConfigSpace(n_regimes=3, n_channels=1, ar_order=2),
        "n_train": 2,
        "n_validation": 1,
        "n_test": 1,
        "seed": 17,
    }

    first = sample_config_split(**kwargs)
    second = sample_config_split(**kwargs)

    assert np.array_equal(first.train[0].transition_matrix, second.train[0].transition_matrix)
    assert np.array_equal(first.test[0].ar_coefficients, second.test[0].ar_coefficients)


def test_switching_ar_config_rejects_non_stochastic_transition_matrix() -> None:
    config = SwitchingARConfig(
        config_id="bad",
        transition_matrix=np.array([[0.5, 0.4], [0.2, 0.8]]),
        ar_coefficients=np.zeros((2, 1, 1)),
        intercepts=np.zeros((2, 1)),
        noise_stds=np.ones((2, 1)),
    )

    with pytest.raises(ValueError, match="rows must sum"):
        config.validate()


def test_generator_space_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="self_transition_range"):
        GeneratorConfigSpace(self_transition_range=(0.9, 1.0)).validate()
