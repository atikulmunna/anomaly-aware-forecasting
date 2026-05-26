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
- `smd_mdn_full_tuned_gpu.csv`: full 28-machine run of the selected-machine MDN tuning winner.
- `smd_joint_tune_gpu.csv`: selected-machine calibrated joint hyperparameter sweep.
- `smd_mdn_tune_gpu.csv`: selected-machine calibrated MDN-LSTM hyperparameter sweep.
- `smd_joint_regime_score_gpu.csv`: selected-machine joint scoring sweep using regime posteriors.
- `smd_joint_pseudo_tune_gpu.csv`: selected-machine joint sweep with train-fitted pseudo-regime labels.
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

The first selected-machine joint tuning sweep did not show a gain from increasing regimes or hidden
size. The calibrated baseline, no-smoothness, and high-smoothness variants tied at F1 `0.2608`.

The selected-machine MDN-LSTM tuning sweep favored the smaller `hidden_size=32` model with F1
`0.2756`, precision `0.3282`, recall `0.2375`, ROC-AUC `0.8333`, and VUS-ROC `0.7496`.
More mixture components and larger hidden size did not improve range F1 in this pass.

When the selected-machine MDN winner was rerun on all 28 SMD machines, performance dropped to F1
`0.2025` with precision `0.1271` and recall `0.4981`. The previous full calibrated MDN-LSTM
(`hidden_size=64`) remains stronger at F1 `0.2310`, so selected-machine tuning is not sufficient
evidence for full-benchmark improvement.

Regime-only joint anomaly scores are currently weak on selected SMD machines. Channel-max forecast
NLL reached F1 `0.2116` in the regime-score sweep, while regime entropy and regime switch scores
were near F1 `0.10`. This indicates the unsupervised regime posterior is not yet a reliable
standalone anomaly score on SMD.

Pseudo-regime supervision helped compared with the same-seed unsupervised reference, but not enough
to beat the best selected-machine joint run. Strong pseudo-regime supervision improved F1 from
`0.2085` to `0.2361`; the earlier selected-machine joint tuning baseline remains higher at
`0.2608`.

## Metric Notes

- `anomaly.threshold_strategy=max_validation_f1` is the original protocol.
- `validation_quantile_99` selects the 99th percentile of validation scores using validation data
  only; the test score distribution is not used for threshold selection.
- `channel_max_nll` computes marginal per-channel NLL and uses the largest channel score.
- `regime_entropy`, `regime_confidence_gap`, and `regime_switch` are joint-only scoring methods
  derived from the regime posterior.
- `window_kmeans` pseudo-regime labels are fitted on training windows only and then applied
  unchanged to validation and test windows.
- Full-scale reports use capped threshold sweeps and a 128-point interval-coverage grid to keep
  evaluation tractable on all 28 SMD machines.
