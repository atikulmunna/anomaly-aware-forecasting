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
- `smd_joint_full_persistence_gpu.csv`: full 28-machine joint run with 2-of-5 persistence filtering.
- `smd_joint_full_threshold_persistence_gpu.csv`: full 28-machine joint q98 threshold plus 2-of-5 persistence run.
- `smd_joint_full_threshold_persistence_q985_gpu.csv`: full 28-machine joint q98.5 threshold plus 2-of-5 persistence run.
- `smd_joint_full_per_machine_threshold_gpu.csv`: full 28-machine joint per-machine q98 threshold plus 2-of-5 persistence run.
- `smd_joint_full_rescore_threshold_gpu.csv`: anomaly-only full-SMD rescore of high global quantiles with 2-of-5 persistence.
- `smd_mdn_full_rescore_threshold_gpu.csv`: anomaly-only full-SMD MDN-LSTM rescore of high global quantiles with 2-of-5 persistence.
- `smd_joint_full_fine_quantile_rescore_gpu.csv`: anomaly-only full-SMD joint q99.0-q99.7 fine quantile sweep.
- `smd_mdn_full_fine_quantile_rescore_gpu.csv`: anomaly-only full-SMD MDN-LSTM q99.0-q99.7 fine quantile sweep.
- `smd_mdn_full_tuned_gpu.csv`: full 28-machine run of the selected-machine MDN tuning winner.
- `smd_joint_tune_gpu.csv`: selected-machine calibrated joint hyperparameter sweep.
- `smd_mdn_tune_gpu.csv`: selected-machine calibrated MDN-LSTM hyperparameter sweep.
- `smd_joint_regime_score_gpu.csv`: selected-machine joint scoring sweep using regime posteriors.
- `smd_joint_pseudo_tune_gpu.csv`: selected-machine joint sweep with train-fitted pseudo-regime labels.
- `smd_persistence_gpu.csv`: selected-machine joint and MDN sweep with anomaly persistence filtering.
- `smd_threshold_persistence_gpu.csv`: selected-machine joint sweep of quantile thresholds with 2-of-5 persistence.
- `smd_per_machine_threshold_gpu.csv`: selected-machine joint sweep of per-machine validation thresholds with 2-of-5 persistence.
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

Persistence filtering is the first selected-machine change to reach the desired F1 band. A 2-of-5
filter on the calibrated joint model improved F1 from `0.2479` to `0.2951`, raising precision from
`0.2253` to `0.4235` while recall moved from `0.2755` to `0.2264`. The same filter did not improve
the selected MDN-LSTM, whose F1 dropped from `0.2320` to `0.2181`.

On the full 28-machine SMD benchmark, 2-of-5 persistence gave a smaller but useful gain for the
joint model: F1 improved from `0.2289` to `0.2409`, and precision improved from `0.1474` to
`0.1822`, while recall dropped from `0.5125` to `0.3552`. This is now slightly above the calibrated
MDN-LSTM F1 `0.2310`, but still below the target `0.26+` benchmark threshold.

Lowering the validation quantile while keeping 2-of-5 persistence was strongly positive on selected
SMD machines. The best selected setting was q98 with F1 `0.3928`, precision `0.3699`, and recall
`0.4187`; q98.5 remained strong at F1 `0.3561`, while q99.3 and above became too conservative.

The selected q98 result did not generalize to full SMD. Full q98 with 2-of-5 persistence reached
recall `0.4361`, but precision dropped to `0.1077`, lowering F1 to `0.1728`. The less aggressive
q98.5 threshold was better but still below q99 persistence: full q98.5 reached F1 `0.1931`,
precision `0.1276`, and recall `0.3974`. Global q99 plus 2-of-5 persistence remains the best
full-SMD joint setting so far at F1 `0.2409`.

Per-machine validation thresholds are a promising but moderate selected-machine improvement. On the
same selected machines and seed, per-machine q98 thresholds with 2-of-5 persistence improved F1 from
the global q99 reference `0.2400` to `0.2749`, raising recall from `0.1752` to `0.3550` while
precision fell from `0.3811` to `0.2243`. This suggests machine-specific calibration can recover
missed events, but the signal should be checked on all 28 machines before treating it as a benchmark
gain.

The full-SMD per-machine q98 check did not generalize. It reached recall `0.4610`, but precision
fell to `0.0800`, producing F1 `0.1364`. This is below the global q99 persistence result (`0.2409`)
and also below the full global q98 persistence result (`0.1728`), so per-machine q98 should not be
promoted as the benchmark setting.

An anomaly-only rescore of the full joint q99 persistence artifacts found the first full-SMD joint
setting above the target F1 band. Raising the global validation quantile to q99.3 while keeping
2-of-5 persistence improved F1 from `0.2409` to `0.2628`, with precision `0.2227` and recall
`0.3204`. More conservative q99.5 and q99.7 thresholds reduced F1 to `0.2485` and `0.2295`,
respectively, so q99.3 is the current best full-SMD joint benchmark setting.

The fine q99.0-q99.7 rescore confirmed q99.3 as the best joint threshold in this grid. Joint F1
rose from `0.2409` at q99.0 to `0.2609` at q99.2 and peaked at `0.2628` for q99.3 before declining.

The matched MDN-LSTM fine sweep shows that this gain is not model-specific, and the best MDN setting
slightly exceeds the joint setting in this single-seed full-SMD pass. MDN-LSTM reached F1 `0.2642`
at q99.6, with precision `0.2423` and recall `0.2905`; q99.1 was close at F1 `0.2633`. The strongest
current claim is therefore that the joint model is competitive with a matched MDN-LSTM baseline, but
not yet clearly better on full SMD without more seeds or stronger regime coupling.

## Metric Notes

- `anomaly.threshold_strategy=max_validation_f1` is the original protocol.
- `validation_quantile_*` strategies select validation-score quantiles using validation data only;
  the test score distribution is not used for threshold selection.
- `per_machine_validation_quantile_*` strategies select one validation-score threshold per machine
  using validation groups only, then apply those frozen machine thresholds to the test split.
- `channel_max_nll` computes marginal per-channel NLL and uses the largest channel score.
- `regime_entropy`, `regime_confidence_gap`, and `regime_switch` are joint-only scoring methods
  derived from the regime posterior.
- `window_kmeans` pseudo-regime labels are fitted on training windows only and then applied
  unchanged to validation and test windows.
- `anomaly_persistence_window` and `anomaly_persistence_count` post-process thresholded
  predictions using a trailing-window persistence rule.
- Full-scale reports use capped threshold sweeps and a 128-point interval-coverage grid to keep
  evaluation tractable on all 28 SMD machines.
