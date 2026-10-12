# Methods

## Dataset and label

The TPEHGDB contains 300 pregnancies represented by 300 uterine EHG recordings.
A term delivery is assigned 0; a preterm delivery is assigned 1.
Source channels S1, S2, S3 correspond to E2−E1, E2−E3, E4−E3.
The filtered signal used here is the source 4th-order digital-filter
0.3–3.0 Hz channel supplied in the WFDB files.

**The recording (not the signal window) is the statistical observation.**
Gestation at delivery, outcome-containing group strings and outcome labels are
not classifier inputs. The optional clinical extension uses only measurements
from the pregnancy's recording time.

## Signal features

Windows of 750 samples (37.5 seconds) are taken on a grid shifted by 125
samples (6.25 seconds). Up to 40 windows at uniformly spaced indices are
examined from each recording. Features are averaged *within recordings*;
models are trained on one row per recording.

For EHG samples `x[0],...,x[N-1]` on a given channel/window:

- **RMS:** `sqrt(mean(x²))`.
- **Variance:** `mean((x-mean(x))²)`.
- **Zero-crossing rate:** `sum(x[t-1] * x[t] < 0) / (N/fs)`.
- **Peak frequency:** `argmax_f PSD(f)` within 0.05–3 Hz.
- **Median frequency:** smallest `f` with cumulative power ≥ half the total
  PSD power within 0.05–3 Hz.
- **Spectral entropy:** `-sum_f q_f log(q_f)` for `q_f = PSD(f)/sum(PSD)`.
- **Symbolic Fourier frequency bins:** normalized count of dominant frequencies
  falling into eight fixed bins over windows. These are lightweight
  bag-of-symbols descriptors inspired by frequency-domain symbolic encoding.

The existing PhysioNet `.fvl` attributes (`RMS`, `Fmed`, `Fpeak`, sample
entropy) are also included. Both these and signal-computed measurements are
available in the generated record matrix. The main five-family experiment
selects `Fmed`, `Fpeak`, `RMS`, computed zero-crossing and computed variance,
each across all three channels.

## Forward feature-family selection

For each outer training fold, start with an empty set `S` of feature families.
At each step, append the family that maximizes mean three-fold inner
cross-validated **balanced accuracy** for a standardized, class-weighted
logistic-regression estimator. Retain an additional family only if the mean
balanced accuracy improves by more than 0.002; use at most five families.
The same selected feature set is used to train the ten baselines and their
ensemble on the corresponding outer train fold.

## Machine learning

All model configurations are defined in `src/tpehg/models.py`. Numeric missing
values are imputed with training-fold medians, then standardized with
training-fold means and variances. Where clinical metadata is included,
nominal missing values are imputed using the train-fold mode and one-hot encoded.

Ten-fold stratified cross-validation uses random seed 2026 and shuffling.
Every fold holds out 30 complete recordings, with exactly one held-out
prediction per recording and classifier. The five-component hard-voting
ensemble uses decision tree, random forest, XGBoost, bagging and gradient
boosting:

`y_hat(x) = mode(C_DT(x), C_RF(x), C_XGB(x), C_BAG(x), C_GB(x))`.

No synthetic records are created before splitting, and preprocessing is fitted
exclusively on training records.

## Neural networks

The deep-learning track uses five deterministic 750-point temporal windows
per recording from the three 0.3–3.0 Hz channels, each converted to 512 time
samples via polyphase resampling and normalized per window. Every fourth
point is retained for model input, forming a 15-channel by 128-sample tensor
for each pregnancy. Five windows remain grouped in one model observation.

A single stratified 75/25 train/test split (seed 2026) is used for neural
networks; no record overlaps partitions. Binary cross entropy with logits is
optimized with Adam using a training-set-only positive-class weight.
Ten two-convolution CNNs vary filter count and kernel size (3 or 5 samples).
The CNN ensemble averages posterior scores from the ten models.
Separate one-layer LSTMs use 20, 50, 100 or 200 memory units.
The sparse autoencoder uses a 16-unit code, reconstruction MSE plus an
L1 activation penalty, followed by a train-only logistic-regression readout.

## Outcome measures

Let TP, TN, FP and FN be the counts in the held-out record-level confusion matrix:

- Accuracy: `(TP+TN)/(TP+TN+FP+FN)`.
- Sensitivity: `TP/(TP+FN)`.
- Specificity: `TN/(TN+FP)`.
- Precision: `TP/(TP+FP)`.
- F1: `2*TP/(2*TP+FP+FN)`.
- Balanced accuracy: `(sensitivity+specificity)/2`.
- ROC AUC: area under ROC computed from continuous scores when defined.

The ML table uses pooled out-of-fold predictions, with per-fold accuracy
standard deviations computed across the ten held-out folds. The neural
network table uses predictions on its separate 75-record test partition.

## Multiband representation and nested model optimization

The feature-augmented experiment uses **all three signal filter configurations**
provided with the TPEHGDB experiments archive (0.3–3, 0.3–4 and 0.08–4 Hz).
For each channel and band, it extracts RMS, median frequency, peak frequency
and sample entropy. Differences and ratios across bands and mean, standard
deviation and range across channels are added without using outcome labels.
Spectral descriptors computed from the input time series remain available.
Gestation at delivery, premature group, the labels and other post-outcome fields
are *never* model inputs. Only the 300 independent pregnancies serve as units.

In each outer fold, a three-fold inner cross-validation chooses model complexity
by ROC AUC. The evaluated learners are balanced random forests, extra trees,
class-weighted radial-basis SVMs, balanced logistic regression, and oversampled
histogram gradient boosting. The oversampler is *inside* the training pipeline;
synthetic / replicated examples from test folds can never reach training.

A second internal out-of-fold prediction sequence chooses the decision
threshold using the training population alone. Define:

\[
 t^* = \arg\max_t \frac12\Big(\frac{TP(t)}{TP(t)+FN(t)}
                         +\frac{TN(t)}{TN(t)+FP(t)}\Big).
\]

The selected threshold is applied unchanged to outer held-out data. Thus the
threshold is not optimized using test outcomes. The calibrated soft-voting
ensemble combines these five learners. Weights use positive contributions from
train-only inner ROC AUC estimates, normalized to sum to one. The soft-voting
ensemble is an *additional* method; the paper's five-tree **hard plurality-voting**
classifier is retained in the primary 10-fold experiment.

The advanced comparison uses the complete outer-fold out-of-fold predictions,
not the fitted scores on training examples. Outer CV uses ten stratified folds,
inner CV three folds, and seed 2026. The parameter grid, threshold and
learner-specific validation scores are saved in `results/advanced/tuning.json`.

## Overlapping-window classification

The paper specifies windows of **750 samples** and **125 samples per shift**.
The window experiment extracts 24 windows per recording, organized as four
well-spaced groups of six consecutive overlapping windows. Each of these
windows is described by all three EHG channels with spectral, RMS,
variance, zero-crossing, entropy, Hjorth mobility and frequency-band power
features. The feature dataset has **7,200 labeled windows from 300 recordings**.

Two clearly distinguished validation targets are evaluated:

- **Window-stratified CV:** train and test contain different window rows, but
  overlapping windows from the **same recording** may occur on both sides.
  This assesses prediction of further windows from recordings partially observed
  during training, not prediction in a new pregnancy.
- **Recording-disjoint CV:** all windows from one pregnancy belong to the same
  fold. This evaluates generalization to completely unseen pregnancies.

Both protocols use ten stratified folds. Random forest, extra trees,
histogram gradient boosting, balanced random forest and 1-nearest-neighbor
matching are included. The input identifier and start sample are **never**
classifier predictors. All prediction rows contain the record identifier,
window index, fold and label.

Because windows of length 750 separated by 125 samples share 625 of 750
samples (83.33%), assigning overlapping windows to independent train/test
partitions changes the independence assumptions of a performance estimate.
`results/windows/overlap_audit.csv` measures the observed train/test sample
reuse in each fold. Each window is a unit of reporting in that experiment,
whereas the primary paper-feature models report **recording-level** performance.

## Interpretation of diagnostic accuracy

The recorded outcomes are 262 term and 38 preterm births. As a result,
classifying every recording as term achieves 262/300 = 87.33% accuracy, but
zero preterm sensitivity. Sensitivity, specificity, balanced accuracy and ROC
AUC accompany all accuracy tables. The trained models and evaluation units
should be considered together; window-level test performance is not
interchangeable with new-pregnancy performance. This code is for research,
not diagnosis or obstetric management.

## Symbolic Fourier bag-of-words representation

For the BOSS-inspired feature experiment, a symbolic Fourier approximation
(SFA) is constructed from the three channels of five fixed waveform segments
per recording. Local length-64 segments shifted by 32 samples are normalized,
and the real and imaginary parts of Fourier coefficients 1 and 2 are retained.
Each coefficient is quantized into four bins using 25/50/75 percentiles
estimated **only from the training recordings of that fold**. The resulting
four-character base-4 words are counted into a per-channel histogram, with
successive duplicate words skipped and word histograms normalized.

The 3 × 256-dimensional representations are evaluated with 1-NN, logistic
regression and random forest under ten-fold recording-disjoint validation.
This implementation uses a transparent SFA-derived bag-of-symbols
representation; exact BOSS variants may differ in distance function,
coefficient selection and numerosity reduction.


## Five-tree window and signal-context model

An additional classifier follows the five-base-learner plurality architecture on
750-sample EHG windows advanced by 125 samples. Twenty-four deterministic
windows are sampled from four regions of each recording. The signal-context
representation concatenates local window statistics and static, label-free
record-level frequency/RMS/variance/spectral descriptors. It never includes
record identifiers, delivery gestational age, term/preterm group labels, or
other outcome-derived attributes as model inputs.

The classifiers are a decision tree, random forest, XGBoost, meta-bagging of
decision trees, and histogram gradient boosting. Each emits a hard class label;
the majority of five labels gives the ensemble prediction. Hyperparameters
and random seeds are fixed in `src/tpehg/context_ensemble.py` and stored in
`results/context_ensemble/config.json`. Imputation is fit inside each fold.

The ten-fold **within-record window** experiment holds out windows, not
recordings, so label-free spectral summaries of the same recording appear
in train and test observations. It evaluates windows from a recording that is
already represented in the training population. The ten-fold
**unseen-recording** experiment uses stratified group folds and no recording
is shared between train and test. Both protocols save individual predictions,
per-fold metrics, and split-overlap counts. These are different estimands.

## Metric aggregation and unit constraints

For any *single* binary confusion matrix with both classes present,

\[\operatorname{Accuracy} = \pi\,\operatorname{Sensitivity}
 +(1-\pi)\,\operatorname{Specificity},\]

where `π` is the prevalence of preterm samples in that test population.
Therefore pooled accuracy must lie between sensitivity and specificity.
Separate studies, averaging methods, and prediction units may produce
statistics that cannot be derived from a single pooled matrix. The
`results/mathematics/metric_geometry.json` file stores the exact algebra and
rounding-grid checks for the study's headline numeric values. All computed
experiment metrics are recalculated directly from saved individual predictions.


## Statistical uncertainty of window metrics

All windows in the same recording share a delivery label and are statistically dependent. Ninety-five percent percentile bootstrap intervals resample **recordings**, not their individual windows. Term and preterm recordings are resampled within their respective strata; every sampled recording contributes all of its window predictions together. This cluster-resampling scheme preserves the study's original class counts and avoids overstating independent sample size from 7,200 correlated windows. The script `scripts/run_intervals.py` writes the results to `results/context_ensemble/cluster_intervals.csv`.
