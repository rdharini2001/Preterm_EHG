#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from tpehg.uncertainty import cluster_bootstrap
out=Path('results/context_ensemble')
b=cluster_bootstrap(pd.read_csv(out/'predictions.csv'))
b.to_csv(out/'cluster_intervals.csv',index=False,float_format='%.8f')
print('Wrote recording-clustered bootstrap confidence intervals',len(b))
