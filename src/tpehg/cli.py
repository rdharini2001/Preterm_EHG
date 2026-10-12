"""Console entrypoints: prepare, evaluate, deep, figures, verify, all."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from .data import build_dataset
from .signals import signal_features
from .evaluation import evaluate_ml,metrics
from .plots import make_plots


def main(argv=None):
    p=argparse.ArgumentParser(description='TPEHGDB EHG ML experiments')
    p.add_argument('command',choices=['prepare','evaluate','deep','figures','verify','all'])
    p.add_argument('--raw-dir',type=Path,default=Path('data/raw/tpehgdb'))
    p.add_argument('--source-dir',type=Path,default=Path('data/source'))
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--clinical',action='store_true',help='Include recording-time clinical variables')
    p.add_argument('--extended',action='store_true',help='Include all derived symbolic/spectral features')
    p.add_argument('--no-forward-selection',action='store_true')
    args=p.parse_args(argv)
    dat=args.data_dir;res=args.results_dir
    if args.command in ('prepare','all'):
        if not args.raw_dir.exists():
            if not (dat/'record_features.csv').exists():
                raise FileNotFoundError('Raw WFDB files unavailable and no prepared dataset found. See README Data setup.')
            print('Using included record-level feature matrix (raw source not present).',flush=True)
        else:
            signal_features(args.raw_dir,dat/'signal_features.csv',dat/'waveforms.npz')
            df=build_dataset(args.source_dir,args.raw_dir,dat/'signal_features.csv')
            dat.mkdir(parents=True,exist_ok=True);df.to_csv(dat/'record_features.csv',index=False)
            print('Prepared',len(df),'records:',int(df.y.sum()),'preterm',flush=True)
    if args.command in ('evaluate','all'):
        df=pd.read_csv(dat/'record_features.csv')
        evaluate_ml(df,res/'tables',kind='all' if args.extended else 'five',clinical=args.clinical,select=not args.no_forward_selection)
        # Persistent predictions for figure and public repository consumers.
        (res/'predictions').mkdir(parents=True,exist_ok=True)
        pd.read_csv(res/'tables/predictions.csv').to_csv(res/'predictions/ml_predictions.csv',index=False)
        make_plots(df,res)
    if args.command=='figures':make_plots(pd.read_csv(dat/'record_features.csv'),res)
    if args.command=='deep':
        from .deep import evaluate_deep
        evaluate_deep(pd.read_csv(dat/'record_features.csv'),dat/'waveforms.npz',res/'tables',epochs=args.epochs)
    if args.command=='all':
        from .deep import evaluate_deep
        if (dat/'waveforms.npz').exists():
            evaluate_deep(pd.read_csv(dat/'record_features.csv'),dat/'waveforms.npz',res/'tables',epochs=args.epochs)
    if args.command in ('verify','all'):
        from .verify import validate_results
        validate_results(dat,res)

if __name__=='__main__':main()
