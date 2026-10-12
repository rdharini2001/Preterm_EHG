"""Validate invariants and all published computational artifacts."""
import numpy as np
import pandas as pd
from .data import feature_columns
from .evaluation import metrics


def validate_results(data_dir,results_dir):
    df=pd.read_csv(data_dir/'record_features.csv')
    assert len(df)==300 and df.Record.nunique()==300 and df.y.sum()==38
    assert not any(f in feature_columns(df,'all') for f in ['Gestation','Premature','Group','Early'])
    p=pd.read_csv(results_dir/'predictions/ml_predictions.csv')
    m=pd.read_csv(results_dir/'tables/model_metrics.csv').set_index('model')
    for name,v in p.groupby('model'):
        assert len(v)==300 and v.Record.nunique()==300
        recalculated=metrics(v.y_true,v.y_pred,v.y_score if v.y_score.notna().all() else None)
        for key in ('accuracy','sensitivity','specificity','f1','balanced_accuracy','precision'):
            assert np.isclose(recalculated[key],m.loc[name,key],atol=5e-8),f'{name}: {key}'
        assert np.isclose(v.y_true.sum(),38)
    split=pd.read_csv(results_dir/'tables/split_manifest.csv')
    assert split.Record.nunique()==300 and (split.groupby('Record').size()==1).all()
    if (results_dir/'tables/deep_predictions.csv').exists():
        d=pd.read_csv(results_dir/'tables/deep_predictions.csv')
        info=pd.read_csv(results_dir/'tables/deep_metrics.csv').set_index('model')
        for name,v in d.groupby('model'):
            calc=metrics(v.y_true,v.y_pred,v.y_score)
            for key in ('accuracy','sensitivity','specificity','f1','auc'):
                assert np.isclose(calc[key],info.loc[name,key],atol=5e-8)
    advanced_dir=results_dir/'advanced'
    if (advanced_dir/'metrics.csv').exists():
        ap=pd.read_csv(advanced_dir/'predictions.csv')
        am=pd.read_csv(advanced_dir/'metrics.csv').set_index('model')
        for name,v in ap.groupby('model'):
            assert len(v)==len(df) and v.Record.nunique()==len(df)
            recalculated=metrics(v.y_true,v.y_pred,v.y_score)
            for key in ('accuracy','sensitivity','specificity','auc'):
                assert np.isclose(recalculated[key],am.loc[name,key],atol=5e-8)
    window_dir=results_dir/'windows'
    if (window_dir/'metrics.csv').exists():
        wp=pd.read_csv(window_dir/'predictions.csv')
        wm=pd.read_csv(window_dir/'metrics.csv').set_index(['evaluation','model'])
        for (protocol,name),v in wp.groupby(['evaluation','model']):
            assert not v.duplicated(['Record','window']).any()
            recalculated=metrics(v.y_true,v.y_pred,v.y_score)
            for key in ('accuracy','sensitivity','specificity','auc'):
                assert np.isclose(recalculated[key],wm.loc[(protocol,name),key],atol=5e-8)
            if protocol=='record_disjoint':
                subjects=[set(t.Record) for _,t in v.groupby('fold')]
                assert not any(a&b for i,a in enumerate(subjects) for b in subjects[i+1:])
    boss_dir=results_dir/'boss'
    if (boss_dir/'metrics.csv').exists():
        bp=pd.read_csv(boss_dir/'predictions.csv')
        bm=pd.read_csv(boss_dir/'metrics.csv').set_index('model')
        for name,v in bp.groupby('model'):
            assert len(v)==len(df) and v.Record.nunique()==len(df)
            calc=metrics(v.y_true,v.y_pred,v.y_score)
            for metric in ('accuracy','sensitivity','specificity','auc'):
                assert np.isclose(calc[metric],bm.loc[name,metric],atol=5e-8)
    print(f'VERIFIED: {len(m)} core ML estimators and advanced, SFA/BOSS, and window protocols; saved predictions agree with results.')
