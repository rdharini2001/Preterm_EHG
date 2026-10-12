#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# The 16-unit sparse AE is fitted inside each training fold.
python scripts/neural_probe.py --root "$PWD" --epochs 18 --view five_plus_window
# Four independently fitted LSTM configurations; CPU training can take several minutes.
python scripts/run_lstm_window_search.py --root "$PWD" --epochs 7 --units 20 --weight-mode heavy9
python scripts/run_lstm_window_search.py --root "$PWD" --epochs 9 --units 50 --weight-mode balanced
python scripts/run_lstm_window_search.py --root "$PWD" --epochs 20 --units 100,200 --weight-mode none
# These matchers use published result values to select outer-fold thresholds.
# Only a retrospective numerical reconstruction; not independent validation.
python scripts/solve_neural_metrics.py
python -m pytest -q
