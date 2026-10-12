# Publish this TPEHG code repository

The GitHub package contains the source-code library, all three source-band
feature tables, 300-record input matrix, 7,200-window cache, five-tree signal-context ensemble results, calibrated-voting protocol search, compact CNN/LSTM
waveforms, model outputs, evaluation reports, figures, tests and CI configuration.
It intentionally excludes the 172 MB original WFDB source archive, which is
included in the separate complete research bundle and is also available
through [PhysioNet](https://physionet.org/content/tpehgdb/1.0.1/).

1. Create an empty GitHub repository on your account, for example `TPEHG-Preterm-ML`.
2. Unzip `tpehg-preterm-neural-methods-github.zip` and open the resulting directory.
3. Run the following commands:

```bash
git init
git add .
git commit -m "Publish electrohysterogram ML code and experiments"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/TPEHG-Preterm-ML.git
git push -u origin main
```

For the original 300 WFDB recordings, use the original archive in the complete
bundle or the public PhysioNet source. Preserve the notices in `docs/DATA.md`
and `docs/UPSTREAM_EXPERIMENTS_LICENSE.txt`.

The documented `scripts/run_protocol_search.sh` targets the uploaded manuscript metrics through a retrospective search. Those selected operating points are exploratory; an independent pregnancy-level test is provided separately.

## Extended computational experiments

The 3-view, 5-protocol study is reproducible from the preprocessed inputs in the GitHub edition:

```bash
pip install -e '.[test,deep]'
bash scripts/run_design_search.sh
```

To re-extract the **180-second transient-trimmed** signal features, first place the licensed/public-source WFDB recordings under `data/raw/tpehgdb/`, then run `REEXTRACT_TRIM=1 bash scripts/run_design_search.sh`. This follows the edge-transient warning in the [official PhysioNet dataset documentation](https://physionet.org/content/tpehgdb/1.0.1/). The pre-CV oversampling and pre-CV SMOTE experiments are intentionally labelled as contamination diagnostics and must not be interpreted as predictions on unseen pregnancies.

## New neural-number experiments

`bash scripts/run_neural_audit.sh` recalculates the saved paired LSTM and AE metrics without training; `bash scripts/run_neural_methods.sh` retrains the associated model variants. The reported matches are post-hoc operating-point selections on overlapping windows and are not prospective predictions on unseen pregnancies. See `docs/NEURAL_METHOD_SEARCH.md`.
