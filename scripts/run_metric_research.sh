#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python scripts/run_window_classifiers.py
python scripts/run_targeted_window_models.py
python -m tpehg.metric_protocols --root "$PWD"
python scripts/make_metric_report.py
python -m pytest -q
