#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP="${1:-data/source/TPEHGDB-Experiments-master.zip}"
if [ ! -f "$ZIP" ]; then
  echo "Supply the TPEHGDB source archive as first argument: $0 path/to/archive.zip" >&2
  exit 1
fi
python scripts/import_source.py "$ZIP"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m tpehg.cli prepare
bash scripts/run_all.sh
