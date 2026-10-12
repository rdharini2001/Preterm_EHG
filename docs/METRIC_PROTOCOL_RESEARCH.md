# Extended metric reconstruction research

All candidates below are calculated from held-out prediction files. This is a forensic, **retrospectively target-ranked** study; its numerical matches do not constitute unbiased estimates on new patients. An accuracy, a class rate, and a fold SD selected from different rows are **not** a single confusion matrix.

**36/75 exact at printed precision; 25/75 additional close; 14/75 still outside stated tolerances.**

Close means at most **0.2 percentage points** for accuracy, **0.005** for sensitivity/specificity, and **0.003** for CV SD. Deep-model RMSE uses the saved probability predictions and was not re-fit or numerically calibrated in this extension.

## Detailed counts

| metric                    |   exact_individual_match |   near_individual_match |   unmatched |
|:--------------------------|-------------------------:|------------------------:|------------:|
| accuracy_percent          |                       15 |                       8 |          11 |
| cv_sd                     |                       11 |                       0 |           0 |
| ensemble_accuracy_percent |                        1 |                       0 |           0 |
| ensemble_cv_sd            |                        1 |                       0 |           0 |
| ensemble_sensitivity      |                        0 |                       1 |           0 |
| ensemble_specificity      |                        1 |                       0 |           0 |
| rmse                      |                        0 |                       1 |           3 |
| sensitivity               |                        3 |                       8 |           0 |
| specificity               |                        4 |                       7 |           0 |

## Paper metric by metric

| location               | model                   | metric                    |   target |   best_computed | status                 |
|:-----------------------|:------------------------|:--------------------------|---------:|----------------:|:-----------------------|
| Abstract / section 3.3 | Ensemble                | ensemble_accuracy_percent |  98.99   |     98.9861     | exact_individual_match |
| Abstract / section 3.3 | Ensemble                | ensemble_sensitivity      |   0.983  |      0.983553   | near_individual_match  |
| Abstract / section 3.3 | Ensemble                | ensemble_specificity      |   0.979  |      0.979008   | exact_individual_match |
| Abstract / section 3.3 | Ensemble                | ensemble_cv_sd            |   0.0113 |      0.0112553  | exact_individual_match |
| Table 1                | kNN                     | accuracy_percent          |  30.1    |     29.8611     | unmatched              |
| Table 1                | kNN                     | cv_sd                     |   0.042  |      0.0419946  | exact_individual_match |
| Table 2                | kNN                     | specificity               |   0.291  |      0.290872   | exact_individual_match |
| Table 2                | kNN                     | sensitivity               |   0.288  |      0.289474   | near_individual_match  |
| Table 4                | kNN                     | accuracy_percent          |  30.1    |     29.8611     | unmatched              |
| Table 1                | Support Vector Machines | accuracy_percent          |  48.3    |     48.1389     | near_individual_match  |
| Table 1                | Support Vector Machines | cv_sd                     |   0.033  |      0.0329401  | exact_individual_match |
| Table 2                | Support Vector Machines | specificity               |   0.333  |      0.333333   | exact_individual_match |
| Table 2                | Support Vector Machines | sensitivity               |   0.431  |      0.430921   | exact_individual_match |
| Table 4                | Support Vector Machines | accuracy_percent          |  48.3    |     48.1389     | near_individual_match  |
| Table 1                | Adaptive Boost          | accuracy_percent          |  70.9    |     71          | near_individual_match  |
| Table 1                | Adaptive Boost          | cv_sd                     |   0.079  |      0.0792324  | exact_individual_match |
| Table 2                | Adaptive Boost          | specificity               |   0.513  |      0.515267   | near_individual_match  |
| Table 2                | Adaptive Boost          | sensitivity               |   0.622  |      0.623904   | near_individual_match  |
| Table 4                | Adaptive Boost          | accuracy_percent          |  70.9    |     71          | near_individual_match  |
| Table 1                | Logistic Regression     | accuracy_percent          |  79.8    |     79.8333     | exact_individual_match |
| Table 1                | Logistic Regression     | cv_sd                     |   0.044  |      0.0442217  | exact_individual_match |
| Table 2                | Logistic Regression     | specificity               |   0.492  |      0.492366   | exact_individual_match |
| Table 2                | Logistic Regression     | sensitivity               |   0.637  |      0.634868   | near_individual_match  |
| Table 4                | Logistic Regression     | accuracy_percent          |  79.8    |     79.8333     | exact_individual_match |
| Table 1                | Naive Bayes             | accuracy_percent          |  86.7    |     86.6667     | exact_individual_match |
| Table 1                | Naive Bayes             | cv_sd                     |   0.026  |      0.0260342  | exact_individual_match |
| Table 2                | Naive Bayes             | specificity               |   0.761  |      0.762405   | near_individual_match  |
| Table 2                | Naive Bayes             | sensitivity               |   0.82   |      0.816886   | near_individual_match  |
| Table 4                | Naive Bayes             | accuracy_percent          |  86.7    |     86.6667     | exact_individual_match |
| Table 1                | Decision Tree           | accuracy_percent          |  94.8    |     94.6667     | near_individual_match  |
| Table 1                | Decision Tree           | cv_sd                     |   0.028  |      0.0281808  | exact_individual_match |
| Table 2                | Decision Tree           | specificity               |   0.925  |      0.923664   | near_individual_match  |
| Table 2                | Decision Tree           | sensitivity               |   0.911  |      0.910088   | near_individual_match  |
| Table 4                | Decision Tree           | accuracy_percent          |  94.8    |     94.6667     | near_individual_match  |
| Table 1                | XG Boost                | accuracy_percent          |  95.9    |     95.9861     | near_individual_match  |
| Table 1                | XG Boost                | cv_sd                     |   0.014  |      0.0140062  | exact_individual_match |
| Table 2                | XG Boost                | specificity               |   0.934  |      0.934796   | near_individual_match  |
| Table 2                | XG Boost                | sensitivity               |   0.901  |      0.900763   | exact_individual_match |
| Table 4                | XG Boost                | accuracy_percent          |  95.9    |     95.9861     | near_individual_match  |
| Table 1                | Meta Bagging Classifier | accuracy_percent          |  96.9    |     96.9306     | exact_individual_match |
| Table 1                | Meta Bagging Classifier | cv_sd                     |   0.014  |      0.0140195  | exact_individual_match |
| Table 2                | Meta Bagging Classifier | specificity               |   0.972  |      0.973282   | near_individual_match  |
| Table 2                | Meta Bagging Classifier | sensitivity               |   0.966  |      0.965649   | exact_individual_match |
| Table 4                | Meta Bagging Classifier | accuracy_percent          |  96.9    |     96.9306     | exact_individual_match |
| Table 1                | Gradient Boost          | accuracy_percent          |  97.4    |     97.3611     | exact_individual_match |
| Table 1                | Gradient Boost          | cv_sd                     |   0.012  |      0.0119388  | exact_individual_match |
| Table 2                | Gradient Boost          | specificity               |   0.97   |      0.969466   | near_individual_match  |
| Table 2                | Gradient Boost          | sensitivity               |   0.952  |      0.950382   | near_individual_match  |
| Table 4                | Gradient Boost          | accuracy_percent          |  97.4    |     97.3611     | exact_individual_match |
| Table 1                | Random Forest           | accuracy_percent          |  97.9    |     97.9008     | exact_individual_match |
| Table 1                | Random Forest           | cv_sd                     |   0.008  |      0.00801341 | exact_individual_match |
| Table 2                | Random Forest           | specificity               |   0.981  |      0.980916   | exact_individual_match |
| Table 2                | Random Forest           | sensitivity               |   0.96   |      0.961832   | near_individual_match  |
| Table 4                | Random Forest           | accuracy_percent          |  97.9    |     97.9008     | exact_individual_match |
| Table 1                | Ensemble Classifier     | accuracy_percent          |  98.9    |     98.855      | exact_individual_match |
| Table 1                | Ensemble Classifier     | cv_sd                     |   0.013  |      0.0127562  | exact_individual_match |
| Table 2                | Ensemble Classifier     | specificity               |   0.979  |      0.977099   | near_individual_match  |
| Table 2                | Ensemble Classifier     | sensitivity               |   0.983  |      0.984733   | near_individual_match  |
| Table 4                | Ensemble Classifier     | accuracy_percent          |  98.9    |     98.855      | exact_individual_match |
| Table 3                | LSTM20                  | accuracy_percent          |  46      |     45.3333     | unmatched              |
| Table 3                | LSTM20                  | rmse                      |   0.51   |      0.488591   | near_individual_match  |
| Table 4                | LSTM20                  | accuracy_percent          |  46      |     45.3333     | unmatched              |
| Table 3                | LSTM50                  | accuracy_percent          |  67      |     66.6667     | exact_individual_match |
| Table 3                | LSTM50                  | rmse                      |   0.43   |      0.465005   | unmatched              |
| Table 4                | LSTM50                  | accuracy_percent          |  67      |     66.6667     | unmatched              |
| Table 3                | LSTM200                 | accuracy_percent          |  73      |     73.3333     | exact_individual_match |
| Table 3                | LSTM200                 | rmse                      |   0.32   |      0.50624    | unmatched              |
| Table 4                | LSTM200                 | accuracy_percent          |  73      |     73.3333     | unmatched              |
| Table 3                | LSTM100                 | accuracy_percent          |  82      |     81.3333     | unmatched              |
| Table 3                | LSTM100                 | rmse                      |   0.24   |      0.535422   | unmatched              |
| Table 4                | LSTM100                 | accuracy_percent          |  82      |     81.3333     | unmatched              |
| Table 4                | Base CNN                | accuracy_percent          |  66.3    |     66.6667     | unmatched              |
| Table 4                | Ensemble DNN            | accuracy_percent          |  72      |     72          | exact_individual_match |
| Table 4                | Sparse AE               | accuracy_percent          |  90.1    |     88          | unmatched              |
| Section 4 narrative    | Sparse AE               | accuracy_percent          |  90.3    |     88          | unmatched              |

## Coherent paired class-rate candidates

Every row below has one dataset, one model, and one classification cutoff. The printed paper accuracy may refer to a different evaluation setting.

| model                   |   target_sensitivity |   sensitivity |   target_specificity |   specificity |   accuracy_percent | source                      | protocol                                                                     |
|:------------------------|---------------------:|--------------:|---------------------:|--------------:|-------------------:|:----------------------------|:-----------------------------------------------------------------------------|
| kNN                     |                0.288 |      0.526316 |                0.291 |      0.324427 |            35      | standard_cv                 | nan                                                                          |
| Support Vector Machines |                0.431 |      0.5      |                0.333 |      0.377863 |            39.3333 | advanced                    | nan                                                                          |
| Adaptive Boost          |                0.622 |      0.623904 |                0.513 |      0.49666  |            51.2778 | window_classifier_extension | evaluation=window_stratified                                                 |
| Logistic Regression     |                0.637 |      0.605263 |                0.492 |      0.48855  |            50.3333 | advanced                    | nan                                                                          |
| Naive Bayes             |                0.82  |      0.473684 |                0.761 |      0.694656 |            66.6667 | standard_cv                 | nan                                                                          |
| Decision Tree           |                0.911 |      0.913377 |                0.925 |      0.916985 |            91.6528 | context_ensemble            | protocol=within_record_windows                                               |
| XG Boost                |                0.901 |      0.908397 |                0.934 |      0.958015 |            93.3206 | design_search               | evaluation=pre_cv_smote_diagnostic|feature_view=three_frequency_bands        |
| Meta Bagging Classifier |                0.966 |      0.969466 |                0.972 |      0.969466 |            96.9466 | design_search               | evaluation=pre_cv_oversampling_diagnostic|feature_view=paper_five            |
| Gradient Boost          |                0.952 |      0.954198 |                0.97  |      0.984733 |            96.9466 | design_search               | evaluation=pre_cv_oversampling_diagnostic|feature_view=three_frequency_bands |
| Random Forest           |                0.96  |      0.961832 |                0.981 |      0.996183 |            97.9008 | design_search               | evaluation=pre_cv_oversampling_diagnostic|feature_view=paper_five            |
| Ensemble Classifier     |                0.983 |      0.977099 |                0.979 |      0.977099 |            97.7099 | design_search               | evaluation=pre_cv_oversampling_diagnostic|feature_view=trimmed_180s          |

## Provenance

- `results/metric_protocols/candidate_metrics.csv` preserves each tested model operating point, population, cutoff, fold SD estimator (sample or population), and TP/TN/FP/FN counts.
- `results/metric_protocols/per_target_candidates.csv` records selected targets and closest predictions.
- `results/window_classifiers/predictions.csv` contains real 10-fold window-stratified and recording-disjoint predictions for logistic regression, linear SVM, Gaussian Naive Bayes, AdaBoost, and two kNN choices.
- `results/window_architecture_sweep/predictions.csv` contains additional 10-fold window-stratified kNN and decision-tree model runs.
- Many high-performing window-stratified protocols have *recording overlap* and cannot establish accuracy for pregnancies not observed during training.
- `results/full_paper_audit/metric_consistency.csv` documents algebraically incompatible same-population paper rows; matching individual metrics cannot remove that constraint.
- The main paper contains both 98.99% (abstract/text) and 98.9% (Table 1), and both 90.3% (text) and 90.1% (Table 4); source values are preserved, not rewritten.

## Re-run

```bash
bash scripts/run_metric_research.sh
```
