"""Recording-clustered bootstrap intervals for correlated EHG window data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def cluster_bootstrap(predictions:pd.DataFrame,n_boot=2000,seed=2026)->pd.DataFrame:
    required={'protocol','model','Record','y_true','y_pred'}
    if not required.issubset(predictions.columns):raise ValueError('Expected per-window predictions')
    results=[]
    for (protocol,model),d in predictions.groupby(['protocol','model'],sort=True):
        grouped=d.groupby('Record',sort=True)
        records=[]
        for rec,g in grouped:
            y=g.y_true.to_numpy(int);p=g.y_pred.to_numpy(int)
            if len(set(y))!=1:raise AssertionError('One pregnancy has contradictory delivery labels')
            records.append([int(y[0]),np.sum((y==1)&(p==1)),np.sum((y==0)&(p==0)),
                            np.sum((y==0)&(p==1)),np.sum((y==1)&(p==0))])
        a=np.asarray(records); positives=a[a[:,0]==1,1:]; negatives=a[a[:,0]==0,1:]
        rng=np.random.default_rng(seed)
        samples_p=positives[rng.integers(len(positives),size=(n_boot,len(positives)))].sum(axis=1)
        samples_n=negatives[rng.integers(len(negatives),size=(n_boot,len(negatives)))].sum(axis=1)
        x=samples_p+samples_n;tp,tn,fp,fn=x.T
        calculated={'accuracy':(tp+tn)/(tp+tn+fp+fn),
                    'sensitivity':tp/np.maximum(1,tp+fn),
                    'specificity':tn/np.maximum(1,tn+fp)}
        for metric,values in calculated.items():
            lo,hi=np.quantile(values,[.025,.975])
            results.append(dict(protocol=protocol,model=model,metric=metric,n_boot=n_boot,
                                cluster='independent EHG recording',lower_95=float(lo),upper_95=float(hi)))
    return pd.DataFrame(results)
