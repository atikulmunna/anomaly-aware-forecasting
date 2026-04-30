# Anomaly-Aware Forecasting

A research prototype for joint probabilistic time series forecasting and regime-switching anomaly detection.

## Overview

This project explores an anomaly-aware forecasting system that predicts future time series values while simultaneously estimating latent regimes and flagging anomalous behavior. The core idea is to avoid treating forecasting and anomaly detection as two disconnected pipelines: the forecasting distribution and regime signal should be learned together and evaluated together.

The planned model family is an MDN-LSTM with a regime head. It produces:

- Probabilistic forecasts using Gaussian mixture outputs
- Posterior probabilities over latent regimes
- Anomaly scores derived from predictive likelihood
- Detection signals for persistent regime shifts and structured sensor failures

## Goals

- Build a reproducible Python research prototype for anomaly-aware forecasting.
- Evaluate probabilistic forecasts using NLL, CRPS, calibration diagnostics, and Energy Score for multivariate outputs.
- Evaluate anomaly detection with rigorous time-series metrics such as range-based precision/recall and VUS-PR/VUS-ROC.
- Use synthetic data with held-out generator configurations for controlled ground-truth evaluation.
- Validate the method on the Server Machine Dataset (SMD) as a real-world benchmark.

## Planned Features

- Synthetic switching time series generator with configurable regimes and failure modes
- SMD ingestion and preprocessing pipeline
- MDN-LSTM forecasting model with diagonal-covariance Gaussian mixture emissions
- Regime posterior estimation and regime-shift detection
- Validation-only threshold selection for reproducible anomaly metrics
- Evaluation harness for forecasting, anomaly detection, and regime detection
- Streamlit dashboard for reviewing archived runs

## Methodological Principles

This project emphasizes disciplined evaluation over leaderboard chasing:

- Test-set thresholds are never used for model or threshold selection.
- Point-adjusted F1 is treated only as a deprecated reference metric.
- Synthetic headline results are reported on held-out generator configurations.
- Standardization statistics are fit on training data only.
- Reported quantitative claims must trace back to reproducible run artifacts.

## Repository Status

This repository is currently being initialized from the project specification. Implementation code, tests, and experiment configuration files will be added incrementally.

## Technology Stack

- Python 3.10+
- PyTorch
- NumPy, SciPy, scikit-learn
- pytest and hypothesis
- ruff and mypy
- Streamlit and Plotly

## License

License information will be added before the first public release.
