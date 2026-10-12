# Experimental numerical coverage

This inventory covers the four printed experimental tables, the headline results, and the extra sparse-autoencoder result in the text. Reported comparisons with other papers are tracked as external literature. It does not purport to regenerate background epidemiology estimates or pixel-derived coordinates in plots.

**Own-study occurrences audited:** 75; 
**individual numerical matches at printed precision:** 22; 
**unmatched occurrences:** 53. 
**Complete four-metric classifier rows from one candidate:** 0/11.

The closest score may come from a different experimental evaluation protocol or a threshold retrospectively selected using known outcomes. The inventory does **not** imply all entries came from a unified trained model.

## Individual numerical targets

| location               | model                   | metric                    |   target |    computed | status          |
|:-----------------------|:------------------------|:--------------------------|---------:|------------:|:----------------|
| Abstract / section 3.3 | Ensemble                | ensemble_accuracy_percent |  98.99   | 98.9861     | numerical_match |
| Abstract / section 3.3 | Ensemble                | ensemble_sensitivity      |   0.983  |  0.983553   | unmatched       |
| Abstract / section 3.3 | Ensemble                | ensemble_specificity      |   0.979  |  0.979008   | numerical_match |
| Abstract / section 3.3 | Ensemble                | ensemble_cv_sd            |   0.0113 |  0.0133333  | unmatched       |
| Table 1                | kNN                     | accuracy_percent          |  30.1    | 35.6667     | unmatched       |
| Table 1                | kNN                     | cv_sd                     |   0.042  |  0.0422953  | numerical_match |
| Table 2                | kNN                     | specificity               |   0.291  |  0.332061   | unmatched       |
| Table 2                | kNN                     | sensitivity               |   0.288  |  0.526316   | unmatched       |
| Table 4                | kNN                     | accuracy_percent          |  30.1    | 35.6667     | unmatched       |
| Table 1                | Support Vector Machines | accuracy_percent          |  48.3    | 48.6667     | unmatched       |
| Table 1                | Support Vector Machines | cv_sd                     |   0.033  |  0.034801   | unmatched       |
| Table 2                | Support Vector Machines | specificity               |   0.333  |  0.40458    | unmatched       |
| Table 2                | Support Vector Machines | sensitivity               |   0.431  |  0.473684   | unmatched       |
| Table 4                | Support Vector Machines | accuracy_percent          |  48.3    | 48.6667     | unmatched       |
| Table 1                | Adaptive Boost          | accuracy_percent          |  70.9    | 71          | unmatched       |
| Table 1                | Adaptive Boost          | cv_sd                     |   0.079  |  0.0792324  | numerical_match |
| Table 2                | Adaptive Boost          | specificity               |   0.513  |  0.492366   | unmatched       |
| Table 2                | Adaptive Boost          | sensitivity               |   0.622  |  0.578947   | unmatched       |
| Table 4                | Adaptive Boost          | accuracy_percent          |  70.9    | 71          | unmatched       |
| Table 1                | Logistic Regression     | accuracy_percent          |  79.8    | 79.6667     | unmatched       |
| Table 1                | Logistic Regression     | cv_sd                     |   0.044  |  0.0442217  | numerical_match |
| Table 2                | Logistic Regression     | specificity               |   0.492  |  0.492366   | numerical_match |
| Table 2                | Logistic Regression     | sensitivity               |   0.637  |  0.605263   | unmatched       |
| Table 4                | Logistic Regression     | accuracy_percent          |  79.8    | 79.6667     | unmatched       |
| Table 1                | Naive Bayes             | accuracy_percent          |  86.7    | 86.6667     | numerical_match |
| Table 1                | Naive Bayes             | cv_sd                     |   0.026  |  0.0266667  | unmatched       |
| Table 2                | Naive Bayes             | specificity               |   0.761  |  0.69084    | unmatched       |
| Table 2                | Naive Bayes             | sensitivity               |   0.82   |  0.473684   | unmatched       |
| Table 4                | Naive Bayes             | accuracy_percent          |  86.7    | 86.6667     | numerical_match |
| Table 1                | Decision Tree           | accuracy_percent          |  94.8    | 94.6667     | unmatched       |
| Table 1                | Decision Tree           | cv_sd                     |   0.028  |  0.0276887  | numerical_match |
| Table 2                | Decision Tree           | specificity               |   0.925  |  0.916985   | unmatched       |
| Table 2                | Decision Tree           | sensitivity               |   0.911  |  0.913377   | unmatched       |
| Table 4                | Decision Tree           | accuracy_percent          |  94.8    | 94.6667     | unmatched       |
| Table 1                | XG Boost                | accuracy_percent          |  95.9    | 96          | unmatched       |
| Table 1                | XG Boost                | cv_sd                     |   0.014  |  0.0138648  | numerical_match |
| Table 2                | XG Boost                | specificity               |   0.934  |  1          | unmatched       |
| Table 2                | XG Boost                | sensitivity               |   0.901  |  0.894737   | unmatched       |
| Table 4                | XG Boost                | accuracy_percent          |  95.9    | 96          | unmatched       |
| Table 1                | Meta Bagging Classifier | accuracy_percent          |  96.9    | 96.9306     | numerical_match |
| Table 1                | Meta Bagging Classifier | cv_sd                     |   0.014  |  0.013662   | numerical_match |
| Table 2                | Meta Bagging Classifier | specificity               |   0.972  |  0.968989   | unmatched       |
| Table 2                | Meta Bagging Classifier | sensitivity               |   0.966  |  0.971491   | unmatched       |
| Table 4                | Meta Bagging Classifier | accuracy_percent          |  96.9    | 96.9306     | numerical_match |
| Table 1                | Gradient Boost          | accuracy_percent          |  97.4    | 97.3611     | numerical_match |
| Table 1                | Gradient Boost          | cv_sd                     |   0.012  |  0.0119388  | numerical_match |
| Table 2                | Gradient Boost          | specificity               |   0.97   |  1          | unmatched       |
| Table 2                | Gradient Boost          | sensitivity               |   0.952  |  0.948465   | unmatched       |
| Table 4                | Gradient Boost          | accuracy_percent          |  97.4    | 97.3611     | numerical_match |
| Table 1                | Random Forest           | accuracy_percent          |  97.9    | 98          | unmatched       |
| Table 1                | Random Forest           | cv_sd                     |   0.008  |  0.00817087 | numerical_match |
| Table 2                | Random Forest           | specificity               |   0.981  |  0.980916   | numerical_match |
| Table 2                | Random Forest           | sensitivity               |   0.96   |  0.947368   | unmatched       |
| Table 4                | Random Forest           | accuracy_percent          |  97.9    | 98          | unmatched       |
| Table 1                | Ensemble Classifier     | accuracy_percent          |  98.9    | 99          | unmatched       |
| Table 1                | Ensemble Classifier     | cv_sd                     |   0.013  |  0.0133333  | numerical_match |
| Table 2                | Ensemble Classifier     | specificity               |   0.979  |  1          | unmatched       |
| Table 2                | Ensemble Classifier     | sensitivity               |   0.983  |  0.990132   | unmatched       |
| Table 4                | Ensemble Classifier     | accuracy_percent          |  98.9    | 99          | unmatched       |
| Table 3                | LSTM20                  | accuracy_percent          |  46      | 46.6667     | unmatched       |
| Table 3                | LSTM20                  | rmse                      |   0.51   |  0.488591   | unmatched       |
| Table 4                | LSTM20                  | accuracy_percent          |  46      | 46.6667     | unmatched       |
| Table 3                | LSTM50                  | accuracy_percent          |  67      | 66.6667     | numerical_match |
| Table 3                | LSTM50                  | rmse                      |   0.43   |  0.465005   | unmatched       |
| Table 4                | LSTM50                  | accuracy_percent          |  67      | 66.6667     | unmatched       |
| Table 3                | LSTM200                 | accuracy_percent          |  73      | 73.3333     | numerical_match |
| Table 3                | LSTM200                 | rmse                      |   0.32   |  0.50624    | unmatched       |
| Table 4                | LSTM200                 | accuracy_percent          |  73      | 73.3333     | unmatched       |
| Table 3                | LSTM100                 | accuracy_percent          |  82      | 81.3333     | unmatched       |
| Table 3                | LSTM100                 | rmse                      |   0.24   |  0.535422   | unmatched       |
| Table 4                | LSTM100                 | accuracy_percent          |  82      | 81.3333     | unmatched       |
| Table 4                | Base CNN                | accuracy_percent          |  66.3    | 66.6667     | unmatched       |
| Table 4                | Ensemble DNN            | accuracy_percent          |  72      | 72          | numerical_match |
| Table 4                | Sparse AE               | accuracy_percent          |  90.1    | 88          | unmatched       |
| Section 4 narrative    | Sparse AE               | accuracy_percent          |  90.3    | 88          | unmatched       |

## Classifier consistency constraints

For any *single binary confusion matrix*, accuracy = prevalence × sensitivity + (1 − prevalence) × specificity. The bounds below allow standard last-digit rounding of the printed values.

| model                   |   paper_accuracy_percent |   allowed_accuracy_min_percent |   allowed_accuracy_max_percent | single_population_feasible_with_rounding   | given_prevalence_feasible_with_rounding   |
|:------------------------|-------------------------:|-------------------------------:|-------------------------------:|:-------------------------------------------|:------------------------------------------|
| kNN                     |                     30.1 |                           28.8 |                           29.1 | False                                      | False                                     |
| Support Vector Machines |                     48.3 |                           33.3 |                           43.1 | False                                      | False                                     |
| Adaptive Boost          |                     70.9 |                           51.3 |                           62.2 | False                                      | False                                     |
| Logistic Regression     |                     79.8 |                           49.2 |                           63.7 | False                                      | False                                     |
| Naive Bayes             |                     86.7 |                           76.1 |                           82   | False                                      | False                                     |
| Decision Tree           |                     94.8 |                           91.1 |                           92.5 | False                                      | False                                     |
| XG Boost                |                     95.9 |                           90.1 |                           93.4 | False                                      | False                                     |
| Meta Bagging Classifier |                     96.9 |                           96.6 |                           97.2 | True                                       | False                                     |
| Gradient Boost          |                     97.4 |                           95.2 |                           97   | False                                      | False                                     |
| Random Forest           |                     97.9 |                           96   |                           98.1 | True                                       | True                                      |
| Ensemble Classifier     |                     98.9 |                           97.9 |                           98.3 | False                                      | False                                     |

## One-protocol joint candidates

| model                   | matched_accuracy   | matched_sensitivity   | matched_specificity   | matched_cv_sd   | same_candidate_all_four_match   |
|:------------------------|:-------------------|:----------------------|:----------------------|:----------------|:--------------------------------|
| kNN                     | False              | False                 | False                 | False           | False                           |
| Support Vector Machines | False              | False                 | False                 | False           | False                           |
| Adaptive Boost          | False              | False                 | False                 | False           | False                           |
| Logistic Regression     | False              | False                 | False                 | False           | False                           |
| Naive Bayes             | False              | False                 | False                 | False           | False                           |
| Decision Tree           | False              | False                 | False                 | False           | False                           |
| XG Boost                | False              | False                 | False                 | False           | False                           |
| Meta Bagging Classifier | True               | False                 | False                 | False           | False                           |
| Gradient Boost          | False              | False                 | False                 | True            | False                           |
| Random Forest           | False              | False                 | False                 | False           | False                           |
| Ensemble Classifier     | False              | False                 | False                 | False           | False                           |

Full numeric files are generated at `results/full_paper_audit/`. The audit never edits model predictions.

## Source-specific differences

- The abstract/section text prints ensemble accuracy **98.99%**; Table 1 gives **98.9%**.
- Section 3.3 prints cross-validation SD **0.0113**; Table 1 gives **0.013**.
- The results narrative says sparse-autoencoder accuracy **90.3%**; Table 4 gives **90.1%**.
- Tables 2 and 4 include published values from other studies, listed independently of the experiments in this repository.
- The paper also specifies **300 signals, 262 term, 38 preterm, 20 Hz, 750-sample windows, 125-sample shift, 6377 initially extracted features, and five selected statistical feature families**. These are methodological/data targets, not fitted metric results.
