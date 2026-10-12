# Evaluation-design experiments

## Companion code and primary research sources

This repository implements EHG signal feature extraction and the five-classifier hard-voting architecture from *Analysis of Electrohysterogram Signals and Prediction of Preterm Births using Machine Learning* (Raghavan et al.; *Biomedical Engineering: Applications, Basis & Communications*, 35(06), 2023, DOI: [10.4015/S1016237223500291](https://doi.org/10.4015/S1016237223500291)). It expands those experiments to directly test the consequences of common sampling and evaluation choices.

**Primary data:** [PhysioNet TPEHGDB v1.0.1](https://physionet.org/content/tpehgdb/1.0.1/), 300 recordings, 262 term and 38 preterm; three EHG channels sampled at 20 Hz. PhysioNet advises ignoring the first and last 180 seconds of filtered signals because of transients. We implement this as a distinct feature view, rather than silently changing the source feature tables.

**Research context:** [Vandewiele et al., EHG Oversampling](https://github.com/GillesVandewiele/EHG-Oversampling) demonstrates that applying SMOTE before data partitioning can produce implausibly high AUC, even on random data. See also [Scientific Data (2023), s41597-023-02581-6](https://www.nature.com/articles/s41597-023-02581-6) and [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).

## Tested design matrix

Three feature views are evaluated independently, with the paper's decision tree, random forest, XGBoost, meta-bagging, gradient boosting, and their five-model hard-voting ensemble:

| Feature view | Construction |
|---|---|
| `paper_five` | Median frequency, peak frequency, RMS, zero crossings, variance; three channels; provided 0.3–3.0 Hz filtered band. |
| `three_frequency_bands` | Three TPEHGDB official filtered bands (0.08–4.0, 0.3–3.0, 0.3–4.0 Hz) and within-record channel/range/ratio descriptors. |
| `trimmed_180s` | Five features again, but recompute waveform-derived zero crossing and variance descriptors after removing 180 seconds at each signal edge. Official `.fvl` entries are retained for the other three features. |

Each feature view uses 10-fold stratified CV and the same model architecture, random seed, and fixed class threshold. The five experiment designs are:

1. `record_cv`: train on nine recording folds, evaluate on independent original pregnancies in the tenth.
2. `train_fold_oversampling`: random minority duplication **inside the training fold only**; independent original pregnancies remain in the validation fold.
3. `train_fold_smote`: SMOTE synthetic minority interpolation **inside the training fold only**; validation contains independent original pregnancies.
4. `pre_cv_oversampling_diagnostic`: minority examples are duplicated **before** fold construction. This can put the identical pregnancy in both training and test folds. **This is a diagnostic of data contamination, not a generalization estimate.**
5. `pre_cv_smote_diagnostic`: imputation and SMOTE interpolation are applied before fold construction. The resulting synthetic examples are derived from the evaluation population as well as training population. **Also a contamination diagnostic, not a generalization estimate.**

All train-only imputation is fit on training recordings, with no delivery-time outcome variable as a predictor. Fixed seeds make the analysis exactly rerunnable. Each model/fold stores `Record`, true and predicted label, predicted score, and feature view. Each fold stores its source-record overlap and synthetic test fraction.

## Mathematical interpretation

For the **same** binary population and pooled confusion matrix, with preterm prevalence π:

\[\mathrm{Accuracy}=\pi\,\mathrm{Sensitivity}+(1-\pi)\,\mathrm{Specificity}.\]

The natural cohort prevalence is 38/300 = 0.1266667. The balanced, pre-split synthetic population has prevalence 1/2. Consequently, equal-looking class-wise rates can correspond to different accuracies across differently balanced cohorts, even without other algorithm changes.

Fold-wise metric aggregation is another independent choice. Let `a_f`, `s_f`, `c_f` be accuracy, sensitivity, and specificity in fold f, and `π_f` the fold's positive prevalence. Then

\[\frac1K\sum_f a_f=\frac1K\sum_f\left[\pi_f s_f+(1-\pi_f)c_f\right].\]

This is **not generally equal** to `(mean π)(mean s) + (1−mean π)(mean c)` if fold composition and class performance covary. For example, fold 1 with 90% preterm and (sensitivity=1, specificity=0), and fold 2 with 10% preterm and (sensitivity=0, specificity=1), each has 90% accuracy, but the two unweighted mean class rates are both 50%. Such arithmetic provides a mathematically coherent explanation of different displayed aggregates **only when the actual evaluation protocol genuinely uses those different averages**.

Cross-validation SD is calculated with population denominator K by default (`np.std(fold_accuracy, ddof=0)`), and SEM is reported separately with `ddof=1` and division by `sqrt(K)`. Replacing one by the other changes the published number; they must not be conflated.

## Observed experimental results

All displayed measurements in this section were computed by fitting the models above. Numbers below come from `results/design_search/metrics.csv`, `results/stability/metrics.csv`, and their associated test predictions. **Diagnostic results include known contamination and must never be represented as independent-pregnancy performance.**

### Three-band model results (percentages for classification metrics)

| evaluation                     | model                |   accuracy |   sensitivity |   specificity |   cv_sd_accuracy |
|:-------------------------------|:---------------------|-----------:|--------------:|--------------:|-----------------:|
| pre_cv_oversampling_diagnostic | Hard Voting Ensemble |      96.56 |        100    |         93.13 |           0.0225 |
| pre_cv_smote_diagnostic        | Hard Voting Ensemble |      92.37 |         96.56 |         88.17 |           0.0421 |
| record_cv                      | Hard Voting Ensemble |      86.67 |          0    |         99.24 |           0.0149 |
| train_fold_oversampling        | Hard Voting Ensemble |      84.67 |         10.53 |         95.42 |           0.0371 |
| train_fold_smote               | Hard Voting Ensemble |      79.67 |         15.79 |         88.93 |           0.0674 |
| pre_cv_oversampling_diagnostic | Random Forest        |      97.71 |        100    |         95.42 |           0.0224 |
| pre_cv_smote_diagnostic        | Random Forest        |      90.84 |         96.56 |         85.11 |           0.0441 |
| record_cv                      | Random Forest        |      87    |          0    |         99.62 |           0.018  |
| train_fold_oversampling        | Random Forest        |      85.33 |          7.89 |         96.56 |           0.0306 |
| train_fold_smote               | Random Forest        |      80    |         18.42 |         88.93 |           0.0907 |

### Three-seed random forest stability

| evaluation                     |   accuracy_mean |   accuracy_min |   accuracy_max |   sensitivity_mean |
|:-------------------------------|----------------:|---------------:|---------------:|-------------------:|
| pre_cv_oversampling_diagnostic |           98.03 |          97.52 |          98.85 |             100    |
| record_cv                      |           86.89 |          86.67 |          87    |               0    |
| train_fold_oversampling        |           84.56 |          83.67 |          85.33 |              12.28 |

### Numerical target coverage

Across the 75 in-study metric occurrences, the previous research branch found 22 isolated numerical matches. Adding the 90 new design/model combinations and three-seed stability analysis changes the combined count to **22**. The new design set does **not** match any complete classifier row. This result is from the tested configurations only, not proof that a more comprehensive method search must fail.

The manuscript also reports values copied from external publications; the code does not relabel those as newly measured metrics.

### Generated artifacts

`results/design_search/metrics.csv`, `fold_metrics.csv`, `predictions.csv`, `split_origins.csv`, `joint_paper_comparison.csv`, `individual_paper_coverage.csv`, `joint_paper_coverage.csv`, `extended_coverage_summary.json`, and `results/stability/` contain the recorded experiments and paper-result coverage. Plots are under `results/figures/design_accuracy.png` and `design_sensitivity.png`.
