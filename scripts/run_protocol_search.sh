#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python scripts/search_published_metrics.py --max-thresholds 101
python scripts/make_protocol_report.py
python scripts/audit_full_paper.py
python -m pytest -q tests/test_protocol_search.py
