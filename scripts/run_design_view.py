#!/usr/bin/env python3
import argparse
from pathlib import Path
from tpehg.data import build_dataset
from tpehg.design_search import evaluate,FEATURE_MODES
p=argparse.ArgumentParser()
p.add_argument('--view',required=True,choices=FEATURE_MODES)
a=p.parse_args()
r=Path(__file__).resolve().parents[1]
f=build_dataset(r/'data/source',r/'data/raw/tpehgdb',r/'data/processed/signal_features.csv')
print(evaluate(f,r/'data/source',r/f'results/design_view_{a.view}',feature_modes=(a.view,)).to_string(index=False))
