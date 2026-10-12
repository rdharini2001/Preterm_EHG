#!/usr/bin/env python3
"""Regenerate 180-second-transient-trimmed features from locally acquired WFDB."""
from pathlib import Path
from tpehg.signals import signal_features
root=Path(__file__).resolve().parents[1]
output=root/'data/processed/signal_features_trim180.csv'
print(f'Extracting: {output}',flush=True)
signal_features(root/'data/raw/tpehgdb',output,trim_seconds=180.0)
