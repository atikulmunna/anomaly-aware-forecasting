# Final Project Summary

This project implemented a reproducible anomaly-aware forecasting prototype for multivariate time
series. The core model is a regime-aware MDN-LSTM that produces probabilistic forecasts, latent
regime posteriors, and anomaly scores from predictive likelihood.

## Final Full-SMD Result

The final benchmark uses the full 28-machine Server Machine Dataset, channel-max predictive NLL,
validation-only quantile thresholds, and 2-of-5 persistence filtering.

| Model | Threshold | Seeds | F1 mean +/- std | Precision mean | Recall mean | ROC-AUC mean | VUS-ROC mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDN-LSTM | q99.6 | 3 | 0.2610 +/- 0.0069 | 0.2387 | 0.2880 | 0.7205 | 0.7137 |
| Joint regime-aware MDN-LSTM | q99.3 | 3 | 0.2446 +/- 0.0209 | 0.1993 | 0.3194 | 0.7144 | 0.7049 |

Conclusion: the joint model is competitive, but the final multi-seed full-SMD benchmark does not
show a clear improvement over the matched non-regime MDN-LSTM baseline.

## What Worked

- Validation-only threshold calibration improved SMD anomaly detection substantially.
- 2-of-5 persistence filtering improved precision without using test-set score distributions.
- The anomaly-only rescore CLI made fair post-processing sweeps much faster.
- Synthetic held-out generator evaluation showed the joint model can help when regime structure is
  controlled and aligned with the data-generating process.

## What Did Not Generalize

- Selected-machine tuning did not reliably transfer to all 28 SMD machines.
- Per-machine q98 thresholds recovered recall on selected machines but over-fired on full SMD.
- Regime-only anomaly scores were weak on SMD.
- The final full-SMD multi-seed result favors the simpler MDN-LSTM baseline.

## Defensible Project Claim

This is best presented as a rigorous research prototype and evaluation harness for anomaly-aware
forecasting, not as a state-of-the-art SMD detector. The value is in the end-to-end implementation,
probabilistic evaluation, validation-only threshold protocol, reproducible suites, and clear negative
findings.

## Key Artifacts

- `reports/smd_final_robustness_gpu.csv`: final three-seed full-SMD robustness results.
- `reports/smd_joint_full_fine_quantile_rescore_gpu.csv`: joint q99.0-q99.7 fine sweep.
- `reports/smd_mdn_full_fine_quantile_rescore_gpu.csv`: MDN q99.0-q99.7 fine sweep.
- `reports/summary.csv`: curated headline result table.
- `experiments/final.smd.robustness.gpu.json`: reproducible final robustness suite.
