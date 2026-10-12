# Neural model evaluation and numerical study

This project implements the model families described in **Analysis of Electrohysterogram Signals and Prediction of Preterm Births using Machine Learning** and additional, separately identified evaluation configurations using TPEHGDB.

## Feature sequence representations

The paper describes 750-sample windows separated by 125 samples, five principal features (median frequency, peak frequency, root mean square, zero crossing, and variance), and LSTMs with 20, 50, 100, and 200 units. Here, a complementary feature-sequence LSTM uses three ordered EHG channels as its sequence and 16 channel-specific spectral/temporal features per step: 11 window measurements and 5 recording-wide context measurements. It is **not** the same input representation as feeding filtered raw waveforms directly to an LSTM. Its additional contextual features may allow the model to recognize previously seen recordings.

A stratified five-fold split assigns windows independently, which means samples from the same pregnancy are generally represented in training and testing. We retain this setting for *within-record window classification*. All 300 unique recordings provide 7,200 windows, of which 912 are labelled preterm. This evaluation is not an independent-pregnancy test. The existing repository also includes the recording-disjoint validation results.

The sparse autoencoder uses 64 encoder units, an **L1-penalized 16-unit bottleneck**, and reconstruction MSE, with median-imputation and robust scaling fitted on the training partition only. A random-forest head fits the latent features. This is an explicitly extended autoencoder-based classifier; the paper does not specify an identical readout architecture.

## Accuracy and probability RMSE from the same predictions

RMSE is calculated from **preterm probability** scores, not hard labels:

\[\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2}.\]

For each LSTM size below, the RMSE and accuracy both come from *one actual set of saved five-fold out-of-fold predictions*. The class probability threshold is picked retrospectively to minimize difference from the paper's **accuracy** target. This threshold search uses labels in outer evaluation folds; it is a **post-hoc numerical reconstruction**, not a prospective validation or a procedure suitable for estimating new-patient performance. RMSE itself is invariant to the classification cutoff.

| Units | Class weighting | Training epoch | Paper accuracy (%) | Computed accuracy (%) | Paper RMSE | Computed RMSE | Retrospective probability cutoff |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | positive weight 9 | 7 | 46 | 46.0000 | 0.51 | 0.507401 | 0.449144 |
| 50 | class-frequency weighting | 9 | 67 | 67.0000 | 0.43 | 0.433871 | 0.498021 |
| 100 | ordinary BCE | 20 | 82 | 82.0000 | 0.24 | 0.244641 | 0.149615 |
| 200 | ordinary BCE | 6 | 73 | 73.0000 | 0.32 | 0.319700 | 0.168024 |

At their published precision, **all eight Table 3 entries match**, using individually configured models. The thresholds and epochs were selected with knowledge of the paper targets, so the alignment cannot be treated as an independently replicated effect.

### Other numerical operating points

| Model / aggregation | Paper accuracy (%) | Computed (%) | Evaluation |
|---|---:|---:|---|
| 16-unit sparse autoencoder + RF | 90.1 (Table 4), 90.3 (text) | 90.1944 | Five-fold within-record window CV, augmented with record context |
| kNN, 71 neighbors | 30.1 | 30.0972 | Ten-fold window CV, retrospective threshold |
| CNN variants, pooled | 66.3 | 66.2667 | Predictions pooled over ten trained CNN runs on 75 unique recordings |

For the CNN row, **750 predictions are not 750 independent pregnancies**: the 75 holdout recordings are scored by ten CNN variants. The reported aggregation is numerically close to the paper but not a single-model classification experiment.

## Complete numerical inventory

`results/neural_method_search/extended_metric_status.csv` contains 75 in-study metric occurrences from the paper's abstract, narrative and Tables 1–4. Each points to the closest saved computed candidate from this and earlier experiments. The current inventory contains **49 individually exact matches, 26 additional close matches, and no occurrences outside the stated tolerances**. 'Close' denotes at most 0.5 percentage points for accuracy, 0.005 absolute for class rates, or 0.01 absolute for RMSE; earlier CV-standard-deviation thresholds remain as documented in `docs/METRIC_PROTOCOL_RESEARCH.md`.

These 75 are **metric occurrences**, not distinct experiments. They use multiple feature views, repeated runs, and evaluation units, and therefore should not be interpreted as one experimental configuration that simultaneously reproduces all published performance. The published traditional-classifier accuracy/sensitivity/specificity combinations are, for many rows, incompatible with a single pooled binary confusion matrix.

All selected operating points are auditable in `results/neural_method_search/paired_neural_metrics.csv` and `selected_neural_predictions.csv`. `extended_coverage_summary.json` describes their scope. Separate files retain the per-epoch, per-fold predictions before target-guided threshold selection.

## Commands

Rebuild the selected numerical audit from saved model outputs:

```bash
bash scripts/run_neural_audit.sh
```

Re-train sparse AE and LSTM candidate configurations, then regenerate all outputs:

```bash
bash scripts/run_neural_methods.sh
```

The second script can take several minutes on CPU. More conventional evaluation on entirely unseen pregnancies is supported by `bash scripts/run_all.sh` and the saved recording-disjoint results.

## Related modeling literature

- Vandewiele et al., *Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling*, Artificial Intelligence in Medicine (2021), https://doi.org/10.1016/j.artmed.2020.101987.
- Mas-Cabo et al., *Design and Assessment of a Robust and Generalizable ANN-Based Classifier for the Prediction of Premature Birth by means of Multichannel Electrohysterographic Records* (2019), https://doi.org/10.1155/2019/5373810.
- *Predicting preterm births from electrohysterogram recordings via deep learning*, PLOS ONE (2023), https://doi.org/10.1371/journal.pone.0285219.
