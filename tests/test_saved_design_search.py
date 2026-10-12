from pathlib import Path
import numpy as np
import pandas as pd
from tpehg.evaluation import metrics
from tpehg.design_coverage import run

R=Path(__file__).resolve().parents[1]

def test_complete_protocol_grid():
    m=pd.read_csv(R/'results/design_search/metrics.csv')
    assert len(m)==3*5*6
    assert set(m.model).__contains__('Hard Voting Ensemble')
    assert m.groupby(['feature_view','evaluation']).model.nunique().eq(6).all()

def test_saved_confusion_matrices():
    m=pd.read_csv(R/'results/design_search/metrics.csv')
    p=pd.read_csv(R/'results/design_search/predictions.csv')
    assert p.groupby(['feature_view','evaluation','model']).size().nunique()==2 # 300 or 524
    for row in m.itertuples():
        d=p[(p.feature_view==row.feature_view)&(p.evaluation==row.evaluation)&(p.model==row.model)]
        score=metrics(d.y_true,d.y_pred,d.y_score)
        for key in ['TN','FP','FN','TP','accuracy','sensitivity','specificity']:
            assert np.isclose(score[key],getattr(row,key))

def test_origin_overlap_is_audited():
    s=pd.read_csv(R/'results/design_search/split_origins.csv')
    no_leak=s[s.evaluation.isin(['record_cv','train_fold_oversampling','train_fold_smote'])]
    assert (no_leak.shared_pregnancies==0).all()
    dup=s[s.evaluation=='pre_cv_oversampling_diagnostic']
    assert (dup.shared_pregnancies>0).all()
    synth=s[s.evaluation=='pre_cv_smote_diagnostic']
    assert (synth.test_synthetic_fraction>0).all()

def test_full_metric_coverage_unchanged():
    summary=run(R)
    assert summary['total_in_study_targets']==75
    assert summary['new_search_full_classifier_rows']==0

def test_seed_stability_saved():
    s=pd.read_csv(R/'results/stability/metrics.csv')
    assert set(s.seed)=={7,42,2026}
    assert s.groupby('evaluation').size().eq(3).all()
    assert (s[(s.evaluation=='pre_cv_oversampling_diagnostic')].sensitivity==1).all()
