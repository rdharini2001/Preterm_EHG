"""Data integrity and score verification for advanced and window protocols."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tpehg.advanced import build_multiband_features, predictors, _optimal_threshold
from tpehg.evaluation import metrics
from tpehg.windows import overlap_audit

ROOT=Path(__file__).resolve().parents[1]


def test_multiband_labels_and_no_delivery_fields():
    d=pd.read_csv(ROOT/'data/processed/record_features.csv')
    all_features=build_multiband_features(d,ROOT/'data/source')
    assert len(all_features)==300
    assert all_features.Record.nunique()==300
    assert all_features.y.sum()==38
    assert set(['Gestation','Group','Early','Premature','Record','y']).isdisjoint(predictors(all_features))
    assert len(predictors(all_features))>len([c for c in d if c not in ['Record','y']])


def test_balanced_threshold_no_test_inputs():
    y=np.array([0,0,0,1,1,1])
    p=np.array([.05,.2,.1,.6,.8,.9])
    threshold,score,se,sp=_optimal_threshold(y,p)
    assert score==1.0
    assert se==1.0 and sp==1.0
    assert 0<=threshold<=1


def test_advanced_saved_predictions():
    path=ROOT/'results/advanced'
    if not (path/'predictions.csv').exists():pytest.skip('Advanced experiment not run')
    p=pd.read_csv(path/'predictions.csv');m=pd.read_csv(path/'metrics.csv').set_index('model')
    for name,g in p.groupby('model'):
        assert len(g)==300 and g.Record.nunique()==300
        assert g.y_true.sum()==38
        assert (g.threshold.between(0,1)).all()
        calc=metrics(g.y_true,g.y_pred,g.y_score)
        for metric in ('accuracy','sensitivity','specificity','balanced_accuracy','auc'):
            assert np.isclose(calc[metric],m.loc[name,metric],atol=5e-8),f'{name}:{metric}'


def test_window_protocol_predictions():
    path=ROOT/'results/windows'
    if not (path/'predictions.csv').exists():pytest.skip('Window experiment not run')
    p=pd.read_csv(path/'predictions.csv');m=pd.read_csv(path/'metrics.csv').set_index(['evaluation','model'])
    for (protocol,name),g in p.groupby(['evaluation','model']):
        assert len(g)==7200
        assert not g.duplicated(['Record','window']).any()
        calc=metrics(g.y_true,g.y_pred,g.y_score)
        for metric in ('accuracy','sensitivity','specificity','balanced_accuracy','auc'):
            assert np.isclose(calc[metric],m.loc[(protocol,name),metric],atol=5e-8)
        if protocol=='record_disjoint':
            sets=[set(t.Record) for f,t in g.groupby('fold')]
            assert not any(sets[a]&sets[b] for a in range(len(sets)) for b in range(a+1,len(sets)))


def test_overlap_audit():
    df=pd.read_csv(ROOT/'data/processed/window_features.csv.gz')
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as t:
        audit=overlap_audit(df,Path(t),folds=5)
        group=audit[audit.evaluation=='record_disjoint']
        assert (group.n_overlapping_with_train==0).all()
        assert (group.n_recordings_shared_train_test==0).all()
        window=audit[audit.evaluation=='window_stratified']
        assert window.pct_overlapping_with_train.gt(50).all()
