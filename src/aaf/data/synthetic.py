"""Synthetic switching-AR time series generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticSeries:
    """Generated synthetic time series and ground-truth labels."""

    observations: FloatArray
    regime_labels: IntArray
    anomaly_labels: IntArray
    config_id: str

    def validate(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError("observations must have shape (T, D)")
        if self.regime_labels.shape != (self.observations.shape[0],):
            raise ValueError("regime_labels must have shape (T,)")
        if self.anomaly_labels.shape != (self.observations.shape[0],):
            raise ValueError("anomaly_labels must have shape (T,)")
        if np.any(~np.isfinite(self.observations)):
            raise ValueError("observations must be finite")


@dataclass(frozen=True)
class SwitchingARConfig:
    """Parameters for a switching autoregressive process."""

    config_id: str
    transition_matrix: FloatArray
    ar_coefficients: FloatArray
    intercepts: FloatArray
    noise_stds: FloatArray
    initial_regime: int = 0

    @property
    def n_regimes(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def ar_order(self) -> int:
        return int(self.ar_coefficients.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.intercepts.shape[1])

    def validate(self) -> None:
        if self.transition_matrix.ndim != 2:
            raise ValueError("transition_matrix must have shape (K, K)")
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("transition_matrix must be square")
        if np.any(self.transition_matrix < 0.0):
            raise ValueError("transition probabilities must be non-negative")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to 1")

        k = self.transition_matrix.shape[0]
        if self.ar_coefficients.ndim != 3:
            raise ValueError("ar_coefficients must have shape (K, p, D)")
        if self.intercepts.ndim != 2 or self.noise_stds.ndim != 2:
            raise ValueError("intercepts and noise_stds must have shape (K, D)")
        if self.ar_coefficients.shape[0] != k:
            raise ValueError("ar_coefficients regime dimension must match transition_matrix")
        if self.intercepts.shape != self.noise_stds.shape:
            raise ValueError("intercepts and noise_stds must have identical shapes")
        if self.intercepts.shape[0] != k:
            raise ValueError("intercepts regime dimension must match transition_matrix")
        if self.ar_coefficients.shape[2] != self.intercepts.shape[1]:
            raise ValueError("ar_coefficients channel dimension must match intercepts")
        if np.any(~np.isfinite(self.ar_coefficients)):
            raise ValueError("ar_coefficients must be finite")
        if np.any(~np.isfinite(self.intercepts)):
            raise ValueError("intercepts must be finite")
        if np.any(~np.isfinite(self.noise_stds)):
            raise ValueError("noise_stds must be finite")
        if np.any(self.noise_stds <= 0.0):
            raise ValueError("noise_stds must be strictly positive")
        if not 0 <= self.initial_regime < k:
            raise ValueError("initial_regime must be a valid regime index")


@dataclass(frozen=True)
class GeneratorConfigSpace:
    """Sampling ranges for held-out synthetic generator configurations."""

    n_regimes: int = 3
    n_channels: int = 1
    ar_order: int = 2
    self_transition_range: tuple[float, float] = (0.985, 0.997)
    coefficient_range: tuple[float, float] = (-0.75, 0.75)
    intercept_range: tuple[float, float] = (-2.0, 2.0)
    noise_std_range: tuple[float, float] = (0.05, 0.5)

    def validate(self) -> None:
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        if self.ar_order < 1:
            raise ValueError("ar_order must be positive")
        _validate_range(self.self_transition_range, "self_transition_range", upper_bound=1.0)
        _validate_range(self.coefficient_range, "coefficient_range")
        _validate_range(self.intercept_range, "intercept_range")
        _validate_range(self.noise_std_range, "noise_std_range", lower_bound=0.0)


@dataclass(frozen=True)
class GeneratorConfigSplit:
    """Generator-level train/validation/test split."""

    train: tuple[SwitchingARConfig, ...]
    validation: tuple[SwitchingARConfig, ...]
    test: tuple[SwitchingARConfig, ...]

    def all_config_ids(self) -> set[str]:
        return {config.config_id for config in self.train + self.validation + self.test}


def generate_switching_ar(
    config: SwitchingARConfig,
    *,
    length: int,
    seed: int,
    burn_in: int = 50,
) -> SyntheticSeries:
    """Generate a switching autoregressive series with regime ground truth."""

    config.validate()
    if length < 1:
        raise ValueError("length must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    rng = np.random.default_rng(seed)
    total_length = length + burn_in
    observations = np.zeros((total_length, config.n_channels), dtype=np.float64)
    regimes = np.empty(total_length, dtype=np.int64)
    regimes[0] = config.initial_regime

    for step in range(1, total_length):
        regimes[step] = rng.choice(config.n_regimes, p=config.transition_matrix[regimes[step - 1]])

    for step in range(total_length):
        regime = regimes[step]
        value = config.intercepts[regime].copy()
        max_lag = min(config.ar_order, step)
        for lag in range(1, max_lag + 1):
            value += config.ar_coefficients[regime, lag - 1] * observations[step - lag]
        observations[step] = value + rng.normal(0.0, config.noise_stds[regime])

    series = SyntheticSeries(
        observations=observations[burn_in:],
        regime_labels=regimes[burn_in:],
        anomaly_labels=np.zeros(length, dtype=np.int64),
        config_id=config.config_id,
    )
    series.validate()
    return series


def sample_switching_ar_config(
    *,
    config_id: str,
    space: GeneratorConfigSpace,
    seed: int,
) -> SwitchingARConfig:
    """Sample one switching-AR configuration from a configuration space."""

    space.validate()
    rng = np.random.default_rng(seed)
    transition_matrix = _sample_transition_matrix(
        rng,
        n_regimes=space.n_regimes,
        self_transition_range=space.self_transition_range,
    )
    ar_coefficients = rng.uniform(
        space.coefficient_range[0],
        space.coefficient_range[1],
        size=(space.n_regimes, space.ar_order, space.n_channels),
    )
    ar_coefficients = _stabilize_ar_coefficients(ar_coefficients)
    intercepts = rng.uniform(
        space.intercept_range[0],
        space.intercept_range[1],
        size=(space.n_regimes, space.n_channels),
    )
    noise_stds = rng.uniform(
        space.noise_std_range[0],
        space.noise_std_range[1],
        size=(space.n_regimes, space.n_channels),
    )
    config = SwitchingARConfig(
        config_id=config_id,
        transition_matrix=transition_matrix,
        ar_coefficients=ar_coefficients,
        intercepts=intercepts,
        noise_stds=noise_stds,
        initial_regime=int(rng.integers(0, space.n_regimes)),
    )
    config.validate()
    return config


def sample_config_split(
    *,
    space: GeneratorConfigSpace,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> GeneratorConfigSplit:
    """Sample disjoint generator configs for train, validation, and held-out test."""

    if n_train < 1:
        raise ValueError("n_train must be positive")
    if n_validation < 1:
        raise ValueError("n_validation must be positive")
    if n_test < 1:
        raise ValueError("n_test must be positive")

    total = n_train + n_validation + n_test
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(total)
    configs = tuple(
        sample_switching_ar_config(
            config_id=f"synthetic-config-{idx:04d}",
            space=space,
            seed=_seed_sequence_to_int(child_seed),
        )
        for idx, child_seed in enumerate(child_seeds)
    )
    return GeneratorConfigSplit(
        train=configs[:n_train],
        validation=configs[n_train : n_train + n_validation],
        test=configs[n_train + n_validation :],
    )


def _sample_transition_matrix(
    rng: np.random.Generator,
    *,
    n_regimes: int,
    self_transition_range: tuple[float, float],
) -> FloatArray:
    matrix = np.zeros((n_regimes, n_regimes), dtype=np.float64)
    for regime in range(n_regimes):
        self_probability = float(rng.uniform(*self_transition_range))
        off_diagonal_mass = 1.0 - self_probability
        if n_regimes == 2:
            off_diagonal = np.ones(1, dtype=np.float64)
        else:
            off_diagonal = rng.dirichlet(np.ones(n_regimes - 1, dtype=np.float64))
        other_regimes = [idx for idx in range(n_regimes) if idx != regime]
        matrix[regime, regime] = self_probability
        matrix[regime, other_regimes] = off_diagonal_mass * off_diagonal
    return matrix


def _stabilize_ar_coefficients(coefficients: FloatArray, *, max_abs_sum: float = 0.95) -> FloatArray:
    abs_sums = np.sum(np.abs(coefficients), axis=1, keepdims=True)
    scale = np.minimum(1.0, max_abs_sum / np.maximum(abs_sums, 1e-12))
    return np.asarray(coefficients * scale, dtype=np.float64)


def _validate_range(
    values: tuple[float, float],
    name: str,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    low, high = values
    if low >= high:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    if lower_bound is not None and low <= lower_bound:
        raise ValueError(f"{name} lower bound must be greater than {lower_bound}")
    if upper_bound is not None and high >= upper_bound:
        raise ValueError(f"{name} upper bound must be less than {upper_bound}")


def _seed_sequence_to_int(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
