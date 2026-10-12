"""Construct recording-disjoint train-fold SFA/BOSS feature word histograms."""
import argparse
from pathlib import Path
import pandas as pd
from tpehg.boss import evaluate_boss
p=argparse.ArgumentParser()
p.add_argument('--folds',type=int,default=10)
p.add_argument('--features',type=Path,default=Path('data/processed/record_features.csv'))
p.add_argument('--waveforms',type=Path,default=Path('data/processed/waveforms.npz'))
p.add_argument('--results-dir',type=Path,default=Path('results/boss'))
a=p.parse_args()
print(evaluate_boss(pd.read_csv(a.features),a.waveforms,a.results_dir,folds=a.folds).to_string(index=False))
