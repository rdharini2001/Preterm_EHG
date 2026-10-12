#!/usr/bin/env python3
"""Run five-tree hard-voting ensemble on overlapping EHG windows."""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tpehg.context_ensemble import run_context_ensemble
p=ArgumentParser();p.add_argument('--records',type=Path,default=Path('data/processed/record_features.csv'))
p.add_argument('--windows',type=Path,default=Path('data/processed/window_features.csv.gz'))
p.add_argument('--output',type=Path,default=Path('results/context_ensemble'))
p.add_argument('--folds',type=int,default=10)
a=p.parse_args()
r=run_context_ensemble(pd.read_csv(a.windows),pd.read_csv(a.records),a.output,folds=a.folds)
print(r[['protocol','model','accuracy','sensitivity','specificity','cv_sd_accuracy']].to_string(index=False))
