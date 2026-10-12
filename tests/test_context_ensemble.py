import json
from pathlib import Path
import numpy as np
import pandas as pd
from tpehg.context_ensemble import prepare_context,trees
from tpehg.metric_geometry import (compatible_metric_triple,implied_accuracy,
                                   nearest_accuracy_counts)

ROOT=Path(__file__).resolve().parents[1]

def test_metric_bounds():
    assert not compatible_metric_triple(.9899,.983,.979)
    assert compatible_metric_triple(.981,.983,.979)
    assert abs(implied_accuracy(.983,.979,38,262)-.9795066666666666)<1e-12
    assert nearest_accuracy_counts(7200,.9899)['nearest_errors']==73

def test_five_paper_models():
    assert list(trees())==['Decision Tree','Random Forest','XGBoost','Meta Bagging','Gradient Boost']

def test_record_context_protects_target():
    a=pd.DataFrame({'Record':['one','two'],'y':[1,0],'Fmed_1':[.2,.7],
                    'Premature':['t','f'],'Gestation':[34,39]})
    w=pd.DataFrame({'Record':['one','one','two'],'y':[1,1,0],'window':[0,1,0],
                    'start_sample':[0,125,0],'Fpeak_1':[.3,.4,.5]})
    z=prepare_context(w,a)
    assert z.context_Fmed_1.tolist()==[.2,.2,.7]
    assert not any(c.startswith('context_Premature') or c.startswith('context_Gestation') for c in z.columns)

def test_saved_context_ensemble_consistency():
    p=ROOT/'results/context_ensemble/predictions.csv';m=ROOT/'results/context_ensemble/metrics.csv'
    if not (p.exists() and m.exists()):return
    data=pd.read_csv(p);summary=pd.read_csv(m)
    for (_,row) in summary.iterrows():
        z=data[(data.protocol==row.protocol)&(data.model==row.model)]
        assert len(z)==7200
        assert z[['Record','window']].duplicated().sum()==0
        assert abs((z.y_true==z.y_pred).mean()-row.accuracy)<1e-7
        assert int(((z.y_true==1)&(z.y_pred==1)).sum())==row.TP
        assert int(((z.y_true==0)&(z.y_pred==0)).sum())==row.TN

def test_cluster_uncertainty():
    from tpehg.uncertainty import cluster_bootstrap
    d=pd.DataFrame({'protocol':['test']*6,'model':['m']*6,'Record':['a','a','b','b','c','c'],
                    'y_true':[1,1,0,0,0,0],'y_pred':[1,1,0,0,0,1]})
    b=cluster_bootstrap(d,n_boot=100)
    assert set(b.metric)=={'accuracy','sensitivity','specificity'}
    assert b.lower_95.le(b.upper_95).all()
