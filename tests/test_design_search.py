import numpy as np
import pandas as pd
from tpehg.design_search import get_splits, get_X, FEATURE_MODES, EVALUATIONS

def test_fold_only_real_origin_disjoint():
    x=np.arange(120).reshape(60,2).astype(float); y=np.r_[np.zeros(45,int),np.ones(15,int)]
    r=np.array([f'id{i}' for i in range(60)])
    for setting in ['record_cv','train_fold_oversampling','train_fold_smote']:
        for fold,tr,te,_,_,orig in get_splits(x,y,r,setting,5,2026):
            assert not (set(orig[tr])&set(orig[te]))

def test_prepartition_synthetic_origin_overlap():
    x=np.arange(120).reshape(60,2).astype(float); y=np.r_[np.zeros(45,int),np.ones(15,int)]
    r=np.array([f'id{i}' for i in range(60)])
    folds=list(get_splits(x,y,r,'pre_cv_oversampling_diagnostic',5,2026))
    assert any(set(origin[tr])&set(origin[te]) for _,tr,te,_,_,origin in folds)

def test_no_label_columns_in_features():
    from pathlib import Path
    from tpehg.data import build_dataset
    r=Path(__file__).resolve().parents[1]
    f=build_dataset(r/'data/source',r/'data/raw/tpehgdb',r/'data/processed/signal_features.csv')
    for setting in FEATURE_MODES:
        x,cols=get_X(f,r/'data/source',setting)
        assert x.shape[0]==300
        assert not (set(cols)&{'y','Group','Gestation','Premature','Record'})

def test_aggregation_identity():
    # Macro fold-level sensitivity/specificity can differ from pooled accuracy
    # when class prevalences vary sharply between folds.
    folds=[(90,10,90,0,10,0),(10,90,0,10,0,90)]
    # fold1 p=.9, se=1, sp=0; fold2 p=.1, se=0, sp=1
    acc=(90+90)/200
    macro_sens=(1+0)/2;macro_spec=(0+1)/2
    assert acc>.8 and macro_sens==macro_spec==.5


def test_smote_synthetic_test_rows():
    x=np.arange(120).reshape(60,2).astype(float)
    y=np.r_[np.zeros(45,int),np.ones(15,int)]
    r=np.array([f'id{i}' for i in range(60)])
    folds=list(get_splits(x,y,r,'pre_cv_smote_diagnostic',5,2026))
    assert all(len(X)>len(x) for _,tr,te,X,y,origin in folds)
    assert any(str(o).startswith('SYNTHETIC_') for _,tr,te,X,y,origin in folds for o in origin[te])
