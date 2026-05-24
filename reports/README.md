# Experiment Reports

This directory contains compact CSV summaries copied out of ignored `runs/` directories.
Raw datasets and full run artifacts are intentionally local-only.

## Main Reports

- `synthetic_smoke.csv`: tiny synthetic sanity runs.
- `synthetic_headline.csv`: held-out synthetic generator benchmark.
- `smd_smoke.csv`: tiny real SMD smoke run on one machine.
- `smd_headline.csv`: selected-machine SMD CPU-era baseline and joint run.
- `smd_headline_gpu.csv`: selected-machine SMD joint GPU rerun.
- `smd_full_gpu.csv`: full 28-machine SMD joint run with original mean-NLL scoring.
- `smd_mdn_full_gpu.csv`: full 28-machine SMD non-regime MDN-LSTM run.
- `smd_score_gpu.csv`: selected-machine comparison of anomaly score aggregations.
- `smd_threshold_gpu.csv`: selected-machine comparison of threshold strategies.
- `smd_full_calibrated_gpu.csv`: full 28-machine calibrated joint and MDN runs.
- `summary.csv`: hand-curated headline comparison table.

## Current Takeaways

The original full-SMD joint run improved over the non-regime MDN baseline on range F1:

- Joint, mean NLL, validation-F1 threshold: `0.1416`
- MDN, mean NLL, validation-F1 threshold: `0.1151`

The larger gain came from threshold calibration. Using channel-max NLL plus the validation 99th
percentile threshold improved full-SMD recall substantially:

- Joint calibrated F1: `0.2289`, recall: `0.5125`
- MDN calibrated F1: `0.2310`, recall: `0.5028`

This suggests the current model family has ranking signal, but validation-F1 thresholding is too
conservative for SMD because the validation split is carved from nominal training data and has no
anomaly labels.

## Metric Notes

- `anomaly.threshold_strategy=max_validation_f1` is the original protocol.
- `validation_quantile_99` selects the 99th percentile of validation scores using validation data
  only; the test score distribution is not used for threshold selection.
- `channel_max_nll` computes marginal per-channel NLL and uses the largest channel score.
- Full-scale reports use capped threshold sweeps and a 128-point interval-coverage grid to keep
  evaluation tractable on all 28 SMD machines.
