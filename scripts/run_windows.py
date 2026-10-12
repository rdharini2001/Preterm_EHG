#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from tpehg.windows import extract_windows,evaluate_windows,overlap_audit
p=argparse.ArgumentParser()
p.add_argument('--raw-dir',type=Path,default=Path('data/raw/tpehgdb'))
p.add_argument('--features',type=Path,default=Path('data/processed/record_features.csv'))
p.add_argument('--cache',type=Path,default=Path('data/processed/window_features.csv.gz'))
p.add_argument('--results-dir',type=Path,default=Path('results/windows'))
p.add_argument('--folds',type=int,default=5)
a=p.parse_args()
if a.cache.exists():d=pd.read_csv(a.cache)
else:
 if not a.raw_dir.exists():raise FileNotFoundError(f'{a.raw_dir} requires raw WFDB files; see README')
 d=extract_windows(a.raw_dir,pd.read_csv(a.features));a.cache.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.cache,index=False,compression='gzip')
print(evaluate_windows(d,a.results_dir,folds=a.folds).to_string(index=False))
print(overlap_audit(d,a.results_dir,folds=a.folds).to_string(index=False))
