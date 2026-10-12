#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python scripts/run_context_ensemble.py --folds 10
python scripts/run_intervals.py
python scripts/analyze_metrics.py
python scripts/plot_context.py
python scripts/make_report.py
python -m pytest -q
