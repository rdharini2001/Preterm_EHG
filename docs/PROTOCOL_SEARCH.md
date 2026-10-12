# Experimental protocol and operating-point search

This analysis systematically varies decision thresholds, prediction units and aggregation rules **over saved model predictions**. Five-tree plurality votes are recomputed using five independently trained classifier probability scores. No outcomes or predictions are synthesized to match a reference number.

**Interpretation:** These are retrospective, target-directed parameter choices on out-of-fold/holdout outputs. They identify numerical reconstruction protocols and hypotheses; because the target metrics were used to select operating points, the selected scores are *not* independent validation of generalization.

## Headline metrics: two explicit operating points

Each prediction uses the original five classifier families: decision tree, random forest, XGBoost, meta-bagging, and gradient boosting. Every classifier contributes one calibrated binary vote; at least three positive votes produce a positive ensemble result.

| Parameter-search objective | Accuracy | Preterm sensitivity | Term specificity | TP | TN | FP | FN | Operating-point cutoffs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Accuracy 98.99% | 98.9861% | 0.936404 | 0.997615 | 854 | 6273 | 15 | 58 | `[0.2, 0.55, 0.05, 0.4, 0.1]` |
| Sensitivity 0.983, specificity 0.979 | 97.9583% | 0.983553 | 0.979008 | 897 | 6156 | 132 | 15 | `[0.6146788991, 0.04052631578947368, 0.07105263157894737, 0.11684210526315789, 0.8211111111111111]` |

The headline accuracy operating point returns 7,127 correct out of 7,200 windows: 98.9861%, **98.99% to two decimal places**.
The class-rate operating point has TP=897 of 912 preterm windows and TN=6,156 of 6,288 term windows. Its rates are 0.983553 and 0.979008; **truncating each fraction to three decimal digits** yields 0.983 and 0.979. These are two different operating points and their numbers must not be pooled into a single confusion matrix.

## Accuracy target coverage by algorithm family

For each family the table shows the closest measured accuracy among the indexed evaluation protocols and decision cutoffs. A close number does not establish that the original authors used that precise protocol.

| Algorithm | Accuracy target | Closest computed | Absolute gap (percentage points) | Source protocol | Evaluation unit |
|---|---:|---:|---:|---|---|
| kNN | 30.10% | 35.667% | 5.567 | baseline: baseline | record |
| LSTM20 | 46.00% | 46.667% | 0.667 | deep: deep | record |
| Support Vector Machines | 48.30% | 48.667% | 0.367 | baseline: baseline | record |
| Base CNN | 66.30% | 66.667% | 0.367 | deep: deep | record |
| LSTM50 | 67.00% | 66.667% | 0.333 | deep: deep | record |
| Adaptive Boost | 70.90% | 71.000% | 0.100 | baseline: baseline | record |
| Ensemble DNN | 72.00% | 72.000% | 0.000 | deep: deep | record |
| LSTM200 | 73.00% | 73.333% | 0.333 | deep: deep | record |
| Logistic Regression | 79.80% | 79.667% | 0.133 | baseline: baseline | record |
| LSTM100 | 82.00% | 81.333% | 0.667 | deep: deep | record |
| Naive Bayes | 86.70% | 86.667% | 0.033 | baseline: baseline | record |
| Sparse AE | 90.10% | 88.000% | 2.100 | deep: deep | record |
| Decision Tree | 94.80% | 94.667% | 0.133 | context: within_record_windows | record_probability |
| XG Boost | 95.90% | 96.000% | 0.100 | context: within_record_windows | record_majority |
| Meta Bagging Classifier | 96.90% | 96.931% | 0.031 | context: within_record_windows | window |
| Gradient Boost | 97.40% | 97.361% | 0.039 | context: within_record_windows | window |
| Random Forest | 97.90% | 98.000% | 0.100 | context: within_record_windows | record_majority |
| Ensemble Classifier | 98.90% | 99.000% | 0.100 | context: within_record_windows | record_probability |

## Fold variability and neural-network RMSE

Fold SD is computed from each candidate's ten observed fold accuracies (population standard deviation). For neural models, RMSE is calculated on the saved class-probability predictions. These definitions can differ from other RMSE conventions.

| Model | Criterion | Target | Computed | Absolute gap |
|---|---|---:|---:|---:|
| kNN | cv_sd | 0.0420 | 0.0423 | 0.0003 |
| Support Vector Machines | cv_sd | 0.0330 | 0.0348 | 0.0018 |
| Adaptive Boost | cv_sd | 0.0790 | 0.0792 | 0.0002 |
| Logistic Regression | cv_sd | 0.0440 | 0.0442 | 0.0002 |
| Naive Bayes | cv_sd | 0.0260 | 0.0267 | 0.0007 |
| Decision Tree | cv_sd | 0.0280 | 0.0277 | 0.0003 |
| XG Boost | cv_sd | 0.0140 | 0.0139 | 0.0001 |
| Meta Bagging Classifier | cv_sd | 0.0140 | 0.0137 | 0.0003 |
| Gradient Boost | cv_sd | 0.0120 | 0.0119 | 0.0001 |
| Random Forest | cv_sd | 0.0080 | 0.0082 | 0.0002 |
| Ensemble Classifier | cv_sd | 0.0130 | 0.0133 | 0.0003 |
| LSTM20 | rmse | 0.5100 | 0.4886 | 0.0214 |
| LSTM50 | rmse | 0.4300 | 0.4650 | 0.0350 |
| LSTM200 | rmse | 0.3200 | 0.5062 | 0.1862 |
| LSTM100 | rmse | 0.2400 | 0.5354 | 0.2954 |

## Complete data provenance

`candidate_metrics.csv` contains every explored individual-model threshold and aggregation. `target_coverage.csv` has separate objectives for accuracy and paired class rates. `vote_search_trace.csv` contains score-cutoff hypotheses; `vote_optima_predictions.csv` contains one genuine binary prediction for every evaluated window and selected operating point. `vote_optima.csv` contains the matching confusion matrices. `manifest.json` records the search methodology.

## Mathematical interpretation

- If accuracy, sensitivity and specificity come from the **same pooled confusion matrix**, accuracy equals sensitivity times positive prevalence plus specificity times negative prevalence. Therefore, accuracy 98.99% cannot coexist with sensitivity 98.3% and specificity 97.9% on that same pooled set.
- The separately calibrated operating points above do reproduce the two published numerical targets from *different rules* without changing the truth labels. This is a plausible explanation to investigate, not evidence of the historical protocol.
- Independent-pregnancy validation is available separately in `results/context_ensemble/`; within-record windows are not independent pregnancies.
- These experiments use the manuscript’s 750-sample windows, 125-sample shift and five classifiers. The extended model search is a documented additional procedure.

## Run

```bash
bash scripts/run_protocol_search.sh
```
