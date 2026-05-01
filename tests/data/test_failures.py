import numpy as np
import pytest

from aaf.data.failures import FailureEvent, apply_failure_events
from aaf.data.synthetic import SyntheticSeries


def make_series() -> SyntheticSeries:
    observations = np.arange(20, dtype=np.float64).reshape(10, 2)
    return SyntheticSeries(
        observations=observations,
        regime_labels=np.zeros(10, dtype=np.int64),
        anomaly_labels=np.zeros(10, dtype=np.int64),
        config_id="cfg",
    )


def test_stuck_at_repeats_previous_value_and_labels_interval() -> None:
    series = apply_failure_events(
        make_series(),
        (FailureEvent(mode="stuck_at", start=3, end=6, channels=(0,)),),
        seed=1,
    )

    assert series.observations[3:6, 0].tolist() == [4.0, 4.0, 4.0]
    assert series.observations[3:6, 1].tolist() == [7.0, 9.0, 11.0]
    assert series.anomaly_labels.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_dropout_sets_segment_to_value() -> None:
    series = apply_failure_events(
        make_series(),
        (FailureEvent(mode="dropout", start=2, end=4, value=-1.0),),
        seed=1,
    )

    assert np.all(series.observations[2:4] == -1.0)


def test_additive_drift_adds_linear_bias() -> None:
    base = make_series()
    series = apply_failure_events(
        base,
        (FailureEvent(mode="additive_drift", start=1, end=4, channels=(1,), magnitude=3.0),),
        seed=1,
    )

    expected = base.observations[1:4, 1] + np.array([0.0, 1.5, 3.0])
    assert series.observations[1:4, 1] == pytest.approx(expected)


def test_multiplicative_noise_is_reproducible_for_seed() -> None:
    event = FailureEvent(mode="multiplicative_noise", start=1, end=8, magnitude=2.0)

    first = apply_failure_events(make_series(), (event,), seed=123)
    second = apply_failure_events(make_series(), (event,), seed=123)

    assert np.array_equal(first.observations, second.observations)
    assert not np.array_equal(first.observations[1:8], make_series().observations[1:8])


def test_spike_train_adds_spikes_at_periodic_indices() -> None:
    base = make_series()
    series = apply_failure_events(
        base,
        (FailureEvent(mode="spike_train", start=1, end=8, channels=(0,), magnitude=10.0, period=3),),
        seed=1,
    )

    changed = np.flatnonzero(series.observations[:, 0] != base.observations[:, 0])
    assert changed.tolist() == [1, 4, 7]
    assert series.observations[changed, 0] == pytest.approx(base.observations[changed, 0] + 10.0)


def test_phase_shift_uses_lagged_values() -> None:
    base = make_series()
    series = apply_failure_events(
        base,
        (FailureEvent(mode="phase_shift", start=3, end=6, channels=(1,), lag=2),),
        seed=1,
    )

    assert series.observations[3:6, 1].tolist() == base.observations[1:4, 1].tolist()


def test_overlapping_events_keep_union_anomaly_labels() -> None:
    series = apply_failure_events(
        make_series(),
        (
            FailureEvent(mode="dropout", start=1, end=4),
            FailureEvent(mode="spike_train", start=3, end=6),
        ),
        seed=1,
    )

    assert series.anomaly_labels.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_rejects_unknown_failure_mode() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        apply_failure_events(make_series(), (FailureEvent(mode="unknown", start=1, end=2),), seed=1)


def test_rejects_out_of_bounds_channel() -> None:
    with pytest.raises(ValueError, match="channel"):
        apply_failure_events(
            make_series(),
            (FailureEvent(mode="dropout", start=1, end=2, channels=(3,)),),
            seed=1,
        )
