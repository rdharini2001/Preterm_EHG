#!/usr/bin/env python
"""Run the pre-registered nested advanced model suite from input feature tables."""
import argparse
from pathlib import Path
import pandas as pd
from tpehg.advanced import evaluate_advanced

p=argparse.ArgumentParser()
p.add_argument('--folds',type=int,default=10)
p.add_argument('--features',type=Path,default=Path('data/processed/record_features.csv'))
p.add_argument('--source-dir',type=Path,default=Path('data/source'))
p.add_argument('--output',type=Path,default=Path('results/advanced'))
a=p.parse_args()
print(evaluate_advanced(pd.read_csv(a.features),a.source_dir,a.output,folds=a.folds).to_string(index=False))
