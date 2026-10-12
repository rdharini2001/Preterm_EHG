#!/usr/bin/env python3
from pathlib import Path
from tpehg.data import build_dataset
from tpehg.design_search import evaluate,paper_search
r=Path(__file__).resolve().parents[1]
f=build_dataset(r/'data/source',r/'data/raw/tpehgdb',r/'data/processed/signal_features.csv')
s=evaluate(f,r/'data/source',r/'results/design_search')
print(s[['feature_view','evaluation','model','accuracy','sensitivity','specificity','cv_sd_accuracy']].to_string(index=False))
print('Closest JOINT candidates:')
print(paper_search(r).to_string(index=False))
