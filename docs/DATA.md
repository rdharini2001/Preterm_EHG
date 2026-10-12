# Data download, schema and licenses

## Original database

[Term–Preterm EHG Database, PhysioNet v1.0.1](https://physionet.org/content/tpehgdb/1.0.1/)
(DOI [10.13026/C2FW2V](https://doi.org/10.13026/C2FW2V)) contains 300
recordings in `.dat` and `.hea` WFDB format; 262 recordings have term-birth
outcomes and 38 have preterm outcomes. The public TPEHGDB database has the
**Open Data Commons Attribution License v1.0**.

The three supplied `.fvl` feature tables originate from the
`TPEHGDB-Experiments` source archive, which is provided for non-commercial,
educational and research use. Please retain the license and appropriate
acknowledgments: [the archive's full license](UPSTREAM_EXPERIMENTS_LICENSE.txt).

## Files in this repository

- `data/source/*.fvl`: source feature tables with three rows per recording.
- `data/processed/signal_features.csv`: computed 750-point-window descriptors.
- `data/processed/record_features.csv`: 300-row matrix for ML evaluation.
- `data/processed/waveforms.npz`: 300 records × 5 windows × 3 channels × 512 samples.
- `data/raw/tpehgdb/`: optional original 300 `.dat` and 300 `.hea` WFDB files.

The supplied source archive can be imported with:

```bash
python scripts/import_source.py /path/to/TPEHGDB-Experiments-master.zip
python -m tpehg.cli prepare
```

You may also download the WFDB records directly from PhysioNet:

```bash
wget -r -N -c -np https://physionet.org/files/tpehgdb/1.0.1/
```

Place all `tpehg*.hea` and `tpehg*.dat` files in `data/raw/tpehgdb/`, and
run `python -m tpehg.cli prepare`. Existing processed matrices allow the
models to run without downloading the complete recordings.

## Sources and feature exclusions

Features derived from `Gestation`, `Group`, `Premature`, or `Early` are never
passed as predictors. The target uses the published `Premature` outcome.
The data matrix includes optional metadata such as `Age`, `Parity` and
`Rectime` when the original WFDB headers are available. Missing metadata
are imputed strictly within model-training folds.

Please cite: Fele-Žorž et al., *Medical & Biological Engineering & Computing*
(2008) and the PhysioNet TPEHGDB archive (v1.0.1, DOI 10.13026/C2FW2V).

## Additional derived inputs

`data/processed/window_features.csv.gz` contains 7,200 labeled 750-sample
window descriptors, derived from the waveform recordings. The public companion
package includes this compact dataset for offline evaluation. To recreate it
from the raw original WFDB files, run `python scripts/run_windows.py` after
using `scripts/import_source.py`; remove or rename the cached `.csv.gz` file
to force fresh signal extraction.

`results/advanced/multiband_features.csv` is assembled on demand from the
three `.fvl` source tables. No table uses `Gestation`, `Premature`, `Group`,
`Early`, or the recording identifier as a model input.


The signal-context ensemble joins label-free recording descriptors to 24 EHG windows per recording by `Record`. These descriptors are derived from full observed recordings, meaning the within-record validation target is continuation-window classification. New-recording experiments hold out all 24 windows and their corresponding summary descriptors together.
