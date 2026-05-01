"""Synthetic sensor-failure injection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aaf.data.synthetic import FloatArray, IntArray, SyntheticSeries

SUPPORTED_FAILURE_MODES = {
    "stuck_at",
    "dropout",
    "additive_drift",
    "multiplicative_noise",
    "spike_train",
    "phase_shift",
}


@dataclass(frozen=True)
class FailureEvent:
    """A sensor-failure event applied to a contiguous interval."""

    mode: str
    start: int
    end: int
    channels: tuple[int, ...] | None = None
    magnitude: float = 1.0
    value: float = 0.0
    period: int = 3
    lag: int = 1

    def validate(self, *, length: int, n_channels: int) -> None:
        if self.mode not in SUPPORTED_FAILURE_MODES:
            raise ValueError(f"unsupported failure mode: {self.mode}")
        if self.start < 0 or self.end <= self.start or self.end > length:
            raise ValueError("failure event must satisfy 0 <= start < end <= length")
        if not np.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite")
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")
        if self.period < 1:
            raise ValueError("period must be positive")
        if self.lag < 1:
            raise ValueError("lag must be positive")
        for channel in selected_channels(self.channels, n_channels):
            if not 0 <= channel < n_channels:
                raise ValueError("failure channel index is out of bounds")


def apply_failure_events(
    series: SyntheticSeries,
    events: tuple[FailureEvent, ...],
    *,
    seed: int,
) -> SyntheticSeries:
    """Return a copy of a synthetic series with sensor failures injected."""

    series.validate()
    rng = np.random.default_rng(seed)
    observations = series.observations.copy()
    anomaly_labels = series.anomaly_labels.copy()
    for event in events:
        event.validate(length=observations.shape[0], n_channels=observations.shape[1])
        channels = selected_channels(event.channels, observations.shape[1])
        _apply_event(observations, event, channels=channels, rng=rng)
        anomaly_labels[event.start : event.end] = 1

    injected = SyntheticSeries(
        observations=observations,
        regime_labels=series.regime_labels.copy(),
        anomaly_labels=anomaly_labels,
        config_id=series.config_id,
    )
    injected.validate()
    return injected


def selected_channels(channels: tuple[int, ...] | None, n_channels: int) -> tuple[int, ...]:
    """Resolve optional channel selection into an explicit tuple."""

    if channels is None:
        return tuple(range(n_channels))
    return channels


def _apply_event(
    observations: FloatArray,
    event: FailureEvent,
    *,
    channels: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    if event.mode == "stuck_at":
        _apply_stuck_at(observations, event, channels)
    elif event.mode == "dropout":
        observations[event.start : event.end, channels] = event.value
    elif event.mode == "additive_drift":
        _apply_additive_drift(observations, event, channels)
    elif event.mode == "multiplicative_noise":
        _apply_multiplicative_noise(observations, event, channels, rng)
    elif event.mode == "spike_train":
        _apply_spike_train(observations, event, channels)
    elif event.mode == "phase_shift":
        _apply_phase_shift(observations, event, channels)
    else:
        raise ValueError(f"unsupported failure mode: {event.mode}")


def _apply_stuck_at(
    observations: FloatArray,
    event: FailureEvent,
    channels: tuple[int, ...],
) -> None:
    if event.start == 0:
        stuck_values = np.full(len(channels), event.value, dtype=np.float64)
    else:
        stuck_values = observations[event.start - 1, channels]
    observations[event.start : event.end, channels] = stuck_values


def _apply_additive_drift(
    observations: FloatArray,
    event: FailureEvent,
    channels: tuple[int, ...],
) -> None:
    drift = np.linspace(0.0, event.magnitude, event.end - event.start, dtype=np.float64)
    observations[event.start : event.end, channels] += drift[:, np.newaxis]


def _apply_multiplicative_noise(
    observations: FloatArray,
    event: FailureEvent,
    channels: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    segment = observations[event.start : event.end, channels]
    scale = np.maximum(np.std(segment, axis=0), 1e-6) * event.magnitude
    observations[event.start : event.end, channels] += rng.normal(
        loc=0.0,
        scale=scale,
        size=segment.shape,
    )


def _apply_spike_train(
    observations: FloatArray,
    event: FailureEvent,
    channels: tuple[int, ...],
) -> None:
    spike_indices = np.arange(event.start, event.end, event.period)
    observations[spike_indices[:, np.newaxis], channels] += event.magnitude


def _apply_phase_shift(
    observations: FloatArray,
    event: FailureEvent,
    channels: tuple[int, ...],
) -> None:
    source_indices = np.maximum(np.arange(event.start, event.end) - event.lag, 0)
    observations[event.start : event.end, channels] = observations[source_indices[:, np.newaxis], channels]
