#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m tpehg.cli all --epochs 10
# Fold-internal hyperparameter tuning and balanced-decision calibration
python scripts/run_advanced.py --folds 10
# Two distinct validation populations from the paper's 750/125 window scheme
python scripts/run_windows.py --folds 10
python scripts/run_boss.py --folds 10
python scripts/run_context_ensemble.py --folds 10
python scripts/run_intervals.py
python scripts/analyze_metrics.py
python scripts/search_published_metrics.py --max-thresholds 101
python scripts/make_protocol_report.py
python scripts/audit_full_paper.py
python scripts/plot_context.py
python scripts/plot_protocols.py
python -m pytest -q
python scripts/plot_deep.py
python scripts/make_report.py
# Extended 3-view, 5-protocol signal and resampling design study
bash scripts/run_design_search.sh
printf '\nCompleted! See docs/RESULTS.md and results/figures/.\n'

# Supplemental classifier families, threshold/aggregation sensitivity, numerical coverage
bash scripts/run_metric_research.sh
