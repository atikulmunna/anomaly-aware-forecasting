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

## Development Workflow

Implementation should proceed module by module, with testing treated as part of completion rather than a later cleanup step.

- Each module must be thoroughly tested after completion and before work moves to the next module.
- New features should not be allowed to break previously completed modules.
- Progress should be committed in small, meaningful increments and pushed regularly.
- Avoid large, end-of-feature commits that mix unrelated changes.
- Keep local-only specification documents and experiment artifacts out of Git.

## Local Development Setup

On Windows, use a project-local virtual environment instead of the global Anaconda environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy scipy pytest mypy ruff hypothesis hatchling
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Run the quality gates from the same environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m ruff check .
```

## Run Diagnostics

Model and baseline pipelines write `mixture_diagnostics.json` beside `metrics.json`.
These diagnostics are intended to catch MDN pathologies early:

- `normalized_entropy_mean` near zero suggests component collapse.
- `active_components_1pct` shows how many components receive meaningful average weight.
- `effective_components` summarizes component usage from mean mixture weights.
- `std` reports predictive standard-deviation health, including tail values.
- `mean_pairwise_distance` helps identify components with identical or near-identical means.

Anomaly reports include both validation-selected operating-point metrics and threshold-free
summaries:

- `threshold` is selected on validation scores by maximizing range-based F1.
- `test` reports range precision, recall, and F1 at the frozen validation threshold.
- `threshold_free.average_precision` and `threshold_free.roc_auc` are pointwise PR/ROC areas.
- `threshold_free.vus_pr` and `threshold_free.vus_roc` are range-aware threshold-sweep areas.

## SMD Pipelines

The SMD pipelines expect the standard directory layout:

```text
SMD/
  train/<machine-id>.txt
  test/<machine-id>.txt
  test_label/<machine-id>.txt
```

Run the seasonal-naive baseline:

```powershell
aaf-smd-baseline C:\path\to\SMD runs\smd_baseline --machine-id machine-1-1
```

Run the joint regime-aware MDN-LSTM:

```powershell
aaf-smd-joint C:\path\to\SMD runs\smd_joint --machine-id machine-1-1 --epochs 5
```

Both commands fit standardization statistics on each machine's training split only, carve validation
from the training tail, freeze threshold selection on validation scores, and write `metrics.json`,
forecast artifacts, anomaly artifacts, and diagnostics into the selected run directory.

## Comparing Runs

Archived run directories can be compared from their `metrics.json` files:

```powershell
aaf-compare-runs runs --output reports/comparison.csv
aaf-compare-runs runs --output reports/comparison.json
```

If a run directory contains `manifest.json`, fields such as `run_id`, `pipeline`, `dataset`, and
`seed` are included as columns. Nested metrics are flattened with dotted names such as
`forecast.nll`, `anomaly.test.f1`, and `anomaly.threshold_free.vus_pr`.

## Technology Stack

- Python 3.10+
- PyTorch
- NumPy, SciPy, scikit-learn
- pytest and hypothesis
- ruff and mypy
- Streamlit and Plotly

## License

License information will be added before the first public release.
