#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
# Each view is independently checkpointed; results are assembled afterward.
# The GitHub package includes the trimmed feature cache. Re-extraction is
# optional and requires the original WFDB recordings acquired from PhysioNet.
if [[ "${REEXTRACT_TRIM:-0}" == "1" || ! -f data/processed/signal_features_trim180.csv ]]; then
    if ! compgen -G 'data/raw/tpehgdb/*.dat' > /dev/null; then
        echo 'Raw TPEHGDB WFDB files are required to regenerate 180s-trimmed features.' >&2
        exit 1
    fi
    python scripts/extract_trimmed_features.py
fi
for view in paper_five three_frequency_bands trimmed_180s; do
    python scripts/run_design_view.py --view "$view"
done
python scripts/combine_design_runs.py
python scripts/run_stability.py
python scripts/audit_design_search.py
python scripts/plot_design_search.py
python -m pytest -q
