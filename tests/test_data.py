from pathlib import Path
import numpy as np
import pandas as pd
from tpehg.data import build_dataset,feature_columns,read_feature_table
from tpehg.evaluation import metrics
from tpehg.models import select_family_groups

HERE=Path(__file__).resolve().parents[1]

def test_source_files_and_labels():
    d=read_feature_table(HERE/'data/source')
    assert d.Record.nunique()==300 and len(d)==900
    assert d.groupby('Record').y.first().sum()==38

def test_record_matrix_and_no_outcome_leakage():
    d=pd.read_csv(HERE/'data/processed/record_features.csv')
    assert len(d)==300 and d.Record.nunique()==300
    cols=feature_columns(d,'five')
    assert len(cols)==15
    assert set(cols).isdisjoint({'Gestation','Premature','Group','Early','Record','y'})

def test_metrics_confusion():
    result=metrics([0,0,1,1],[0,1,0,1],[.1,.7,.3,.9])
    assert result['accuracy']==.5
    assert result['sensitivity']==.5
    assert result['specificity']==.5
    assert result['TP']==result['TN']==1

def test_forward_selection_uses_expected_feature_families():
    d=pd.read_csv(HERE/'data/processed/record_features.csv')
    X=d[feature_columns(d,'five')]
    cols,groups,scores=select_family_groups(X,d.y,tuple(X.columns),max_groups=2)
    assert len(groups)>=1
    assert all(sum(c.startswith(g+'_') for c in cols)==3 for g in groups)
    assert all(0<=s<=1 for s in scores)

def test_no_overlapping_cv_record_ids():
    p=HERE/'results/tables/split_manifest.csv'
    if p.exists():
        df=pd.read_csv(p)
        assert df.Record.nunique()==len(df)==300

def test_raw_waveform_shape():
    p=HERE/'data/processed/waveforms.npz'
    with np.load(p) as z:
        assert z['waves'].shape==(300,5,3,512)
        assert len(set(z['record']))==300
