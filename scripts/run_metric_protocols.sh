#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m tpehg.metric_protocols --root "$PWD"
python -m pytest -q tests/test_metric_protocols.py
